/* Prefetch rings with bandwidth and latency emulation per tier. */
#define _POSIX_C_SOURCE 200809L
#include "tern_io.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const TernMat *M;
    int seg, r, n;
    size_t bytes;
    const uint8_t *src;
    off_t off;
} Chunk;

typedef struct {
    int used, tier;
    double bw, lat;
    Chunk *sched;
    size_t nsched, cap;
    uint8_t *ring;
    size_t slot;
    int nslots, head, tail, count;
    size_t *meta;
    double *t_free;
    size_t ci;
    pthread_mutex_t mu;
    pthread_cond_t cv_full, cv_empty;
    pthread_t th;
    int stop, fd, err;
    double wait_s;
} Chan;

struct TernIO {
    Chan ch[TERN_NTIERS];
    int mismatch;
};

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void sleep_until(double t)
{
    struct timespec ts;
    ts.tv_sec = (time_t)t;
    ts.tv_nsec = (long)((t - (double)ts.tv_sec) * 1e9);
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR)
        ;
}

static void plan_cb(void *ctx, const TernMat *M, int seg, int r, int n)
{
    TernIO *io = (TernIO *)ctx;
    const TernSeg *g = &M->seg[seg];
    Chan *c = &io->ch[g->tier];
    if (c->nsched == c->cap) {
        size_t nc = c->cap ? 2 * c->cap : 1024;
        Chunk *ns = (Chunk *)realloc(c->sched, nc * sizeof(Chunk));
        if (!ns) {
            c->err = 1;
            return;
        }
        c->sched = ns;
        c->cap = nc;
    }
    Chunk *k = &c->sched[c->nsched++];
    k->M = M;
    k->seg = seg;
    k->r = r;
    k->n = n;
    k->bytes = (size_t)n * (size_t)M->row_bytes;
    k->src = g->data ? g->data + (size_t)(r - g->r0) * (size_t)M->row_bytes : NULL;
    k->off = (off_t)(g->off + (long)((size_t)(r - g->r0) * (size_t)M->row_bytes));
    c->used = 1;
}

static void *producer(void *arg)
{
    Chan *c = (Chan *)arg;
    size_t idx = 0;
    double deadline = now_s();
    for (;;) {
        pthread_mutex_lock(&c->mu);
        while (c->count == c->nslots && !c->stop)
            pthread_cond_wait(&c->cv_empty, &c->mu);
        if (c->stop) {
            pthread_mutex_unlock(&c->mu);
            break;
        }
        const int slot = c->head;
        const double tfree = c->t_free[slot];
        pthread_mutex_unlock(&c->mu);

        const Chunk *k = &c->sched[idx % c->nsched];
        uint8_t *dst = c->ring + (size_t)slot * c->slot;
        if (k->src) {
            memcpy(dst, k->src, k->bytes);
        } else if (pread(c->fd, dst, k->bytes, k->off) != (ssize_t)k->bytes) {
            memset(dst, 0, k->bytes);
            c->err = 1;
        }
        if (c->bw > 0.0) {
            double start = deadline > tfree ? deadline : tfree;
            deadline = start + c->lat + (double)k->bytes / c->bw;
            sleep_until(deadline);
        }
        pthread_mutex_lock(&c->mu);
        c->meta[slot] = idx % c->nsched;
        c->head = (c->head + 1) % c->nslots;
        c->count++;
        pthread_cond_signal(&c->cv_full);
        pthread_mutex_unlock(&c->mu);
        idx++;
    }
    return NULL;
}

static const uint8_t *io_fetch(void *vio, const TernMat *M, int seg, int r, int n)
{
    TernIO *io = (TernIO *)vio;
    Chan *c = &io->ch[M->seg[seg].tier];
    pthread_mutex_lock(&c->mu);
    if (c->count == 0) {
        const double t0 = now_s();
        while (c->count == 0)
            pthread_cond_wait(&c->cv_full, &c->mu);
        c->wait_s += now_s() - t0;
    }
    const int slot = c->tail;
    const Chunk *k = &c->sched[c->meta[slot]];
    if (c->meta[slot] != c->ci || k->M != M || k->seg != seg || k->r != r || k->n != n)
        io->mismatch = 1;
    c->ci = (c->ci + 1) % c->nsched;
    pthread_mutex_unlock(&c->mu);
    return c->ring + (size_t)slot * c->slot;
}

static void io_release(void *vio, int tier)
{
    TernIO *io = (TernIO *)vio;
    Chan *c = &io->ch[tier];
    pthread_mutex_lock(&c->mu);
    c->t_free[c->tail] = now_s();
    c->tail = (c->tail + 1) % c->nslots;
    c->count--;
    pthread_cond_signal(&c->cv_empty);
    pthread_mutex_unlock(&c->mu);
}

TernIO *tern_io_start(const TernModel *m, TernState *s0, const char *path,
                      const double bw[TERN_NTIERS], const double lat[TERN_NTIERS],
                      const size_t ring_bytes[TERN_NTIERS], char *err, size_t errlen)
{
    TernIO *io = (TernIO *)calloc(1, sizeof(TernIO));
    if (!io) {
        snprintf(err, errlen, "out of memory");
        return NULL;
    }
    tern_plan(m, s0->chunk_bytes, plan_cb, io);
    for (int t = 0; t < TERN_NTIERS; t++) {
        Chan *c = &io->ch[t];
        c->tier = t;
        c->fd = -1;
        if (c->err) {
            snprintf(err, errlen, "out of memory for schedule");
            tern_io_stop(io);
            return NULL;
        }
        if (!c->used)
            continue;
        size_t slot = 0;
        for (size_t i = 0; i < c->nsched; i++)
            if (c->sched[i].bytes > slot)
                slot = c->sched[i].bytes;
        slot = (slot + 15u) & ~(size_t)15u;
        c->slot = slot;
        c->nslots = (int)(ring_bytes[t] / slot);
        if (c->nslots < 1)
            c->nslots = 1;
        c->bw = bw[t];
        c->lat = lat[t];
        c->ring = (uint8_t *)malloc((size_t)c->nslots * slot);
        c->meta = (size_t *)calloc((size_t)c->nslots, sizeof(size_t));
        c->t_free = (double *)calloc((size_t)c->nslots, sizeof(double));
        if (!c->ring || !c->meta || !c->t_free) {
            snprintf(err, errlen, "out of memory for %s ring", tern_tier_name(t));
            tern_io_stop(io);
            return NULL;
        }
        if (m->placed[t] > 0 && t == TERN_TIER_SD) {
            c->fd = open(path, O_RDONLY);
            if (c->fd < 0) {
                snprintf(err, errlen, "cannot open %s", path);
                tern_io_stop(io);
                return NULL;
            }
        }
        pthread_mutex_init(&c->mu, NULL);
        pthread_cond_init(&c->cv_full, NULL);
        pthread_cond_init(&c->cv_empty, NULL);
        if (pthread_create(&c->th, NULL, producer, c) != 0) {
            snprintf(err, errlen, "cannot create %s thread", tern_tier_name(t));
            c->used = 0;
            tern_io_stop(io);
            return NULL;
        }
    }
    s0->io = io;
    s0->io_fetch = io_fetch;
    s0->io_release = io_release;
    return io;
}

void tern_io_stats(const TernIO *io, double wait_s[TERN_NTIERS], int *mismatch)
{
    for (int t = 0; t < TERN_NTIERS; t++)
        wait_s[t] = io->ch[t].wait_s;
    *mismatch = io->mismatch;
}

void tern_io_reset_stats(TernIO *io)
{
    for (int t = 0; t < TERN_NTIERS; t++)
        io->ch[t].wait_s = 0.0;
}

int tern_io_slots(const TernIO *io, int tier)
{
    return io->ch[tier].used ? io->ch[tier].nslots : 0;
}

void tern_io_stop(TernIO *io)
{
    if (!io)
        return;
    for (int t = 0; t < TERN_NTIERS; t++) {
        Chan *c = &io->ch[t];
        if (c->used && c->ring && c->meta && c->t_free) {
            pthread_mutex_lock(&c->mu);
            c->stop = 1;
            pthread_cond_broadcast(&c->cv_empty);
            pthread_mutex_unlock(&c->mu);
            pthread_join(c->th, NULL);
            pthread_mutex_destroy(&c->mu);
            pthread_cond_destroy(&c->cv_full);
            pthread_cond_destroy(&c->cv_empty);
        }
        if (c->fd >= 0)
            close(c->fd);
        free(c->sched);
        free(c->ring);
        free(c->meta);
        free(c->t_free);
    }
    free(io);
}
