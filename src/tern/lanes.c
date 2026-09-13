/* Multi-lane passes: N sequences share one read of the weights. */
#define _POSIX_C_SOURCE 200809L
#include "tern.h"
#include "tern_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int cmd_lanes(int argc, char **argv);

static double wall(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int argmax_f(const float *x, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b])
            b = i;
    return b;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(double), cmp_d);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static int parse_tiers(const char *s, double *v, double scale)
{
    char buf[256];
    strncpy(buf, s, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        char *eq = strchr(tok, '=');
        if (!eq)
            return -1;
        *eq = '\0';
        int t = -1;
        for (int k = 0; k < TERN_NTIERS; k++)
            if (strcmp(tok, tern_tier_name(k)) == 0)
                t = k;
        if (t < 0)
            return -1;
        v[t] = strcmp(eq + 1, "inf") == 0 ? 0.0 : atof(eq + 1) * scale;
    }
    return 0;
}

int cmd_lanes(int argc, char **argv)
{
    const char *path = argv[2], *place = NULL;
    int nl = 1, steps = 3, kv = TERN_KV_F32, sync = 1, check = 0;
    size_t chunk = TERN_CHUNK_DEFAULT, ring = 256u * 1024u;
    double bw[TERN_NTIERS] = {0, 0, 0}, lat[TERN_NTIERS] = {0, 0, 0};
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (strcmp(a, "--sync") == 0) { sync = 1; continue; }
        if (strcmp(a, "--check") == 0) { check = 1; continue; }
        if (!v) { fprintf(stderr, "missing value for %s\n", a); return 2; }
        if (strcmp(a, "--lanes") == 0) nl = atoi(v);
        else if (strcmp(a, "--steps") == 0) steps = atoi(v);
        else if (strcmp(a, "--place") == 0) place = v;
        else if (strcmp(a, "--kv") == 0) kv = strcmp(v, "int8") == 0 ? TERN_KV_I8 : TERN_KV_F32;
        else if (strcmp(a, "--chunk") == 0) chunk = (size_t)atol(v) * 1024u;
        else if (strcmp(a, "--ring") == 0) ring = (size_t)atol(v) * 1024u;
        else if (strcmp(a, "--bw") == 0) {
            if (parse_tiers(v, bw, 1e6) != 0) { fprintf(stderr, "malformed --bw\n"); return 2; }
            sync = 0;
        } else if (strcmp(a, "--lat") == 0) {
            if (parse_tiers(v, lat, 1e-3) != 0) { fprintf(stderr, "malformed --lat\n"); return 2; }
        } else { fprintf(stderr, "unknown option: %s\n", a); return 2; }
        i++;
    }
    if (nl < 1 || nl > TERN_MAX_LANES || steps < 1) {
        fprintf(stderr, "lanes 1..%d and steps >= 1\n", TERN_MAX_LANES);
        return 2;
    }

    for (int i = 3; i < argc; i++)
        if (strcmp(argv[i], "--sync") == 0)
            sync = 1;

    TernModel m;
    char err[256];
    if (tern_open(&m, path, place, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 2;
    }
    if (steps + 1 > m.cfg.max_seq) {
        fprintf(stderr, "error: steps + 1 > max_seq (%d)\n", m.cfg.max_seq);
        return 2;
    }
    TernState *st = (TernState *)calloc((size_t)nl, sizeof(TernState));
    TernState *L[TERN_MAX_LANES];
    for (int i = 0; i < nl; i++) {
        if (tern_state_init(&st[i], &m, kv, 0) != 0) {
            fprintf(stderr, "error: out of memory for lane %d\n", i);
            return 2;
        }
        L[i] = &st[i];
    }
    tern_set_chunk(L[0], chunk);
    TernIO *io = NULL;
    if (!sync) {
        const size_t rings[TERN_NTIERS] = {ring, ring, ring};
        io = tern_io_start(&m, L[0], path, bw, lat, rings, err, sizeof err);
        if (!io) {
            fprintf(stderr, "error: %s\n", err);
            return 2;
        }
    }

    int *toks = (int *)malloc(sizeof(int) * (size_t)nl * (size_t)(steps + 1));
    for (int i = 0; i < nl; i++)
        toks[i] = (i * 7919 + 1) % m.cfg.vocab;
    double *tw = (double *)malloc(sizeof(double) * (size_t)steps);
    double *tws = (double *)malloc(sizeof(double) * (size_t)steps);
    double prev_wait = 0.0;

    for (int p = 0; p <= steps; p++) {
        if (p == 1) {
            tern_state_reset_counters(L[0]);
            if (io) {
                tern_io_reset_stats(io);
                prev_wait = 0.0;
            }
        }
        const double t0 = wall();
        tern_forward_batch(&m, L, nl, toks + (size_t)p * nl, p);
        const double dt = wall() - t0;
        if (p < steps)
            for (int i = 0; i < nl; i++)
                toks[(size_t)(p + 1) * nl + i] = argmax_f(L[i]->logits, m.cfg.vocab);
        if (p >= 1) {
            tw[p - 1] = dt;
            double w[TERN_NTIERS] = {0, 0, 0}, tot = 0.0;
            int mm = 0;
            if (io) {
                tern_io_stats(io, w, &mm);
                for (int t = 0; t < TERN_NTIERS; t++)
                    tot += w[t];
            }
            tws[p - 1] = tot - prev_wait;
            prev_wait = tot;
        }
    }
    double w[TERN_NTIERS] = {0, 0, 0};
    int mismatch = 0;
    if (io)
        tern_io_stats(io, w, &mismatch);
    uint64_t bytes = 0;
    for (int t = 0; t < TERN_NTIERS; t++)
        bytes += L[0]->tier_bytes[t];
    const double bytes_pass = (double)bytes / steps;
    const double macs_lane = (double)L[0]->macs / steps / nl;
    const double wall_med = median(tw, steps), wait_med = median(tws, steps);

    int ok = 1;
    if (check) {
        TernModel m2;
        if (tern_open(&m2, path, place, err, sizeof err) != 0) {
            fprintf(stderr, "error: %s\n", err);
            return 2;
        }
        for (int i = 0; i < nl && ok; i++) {
            TernState s;
            tern_state_init(&s, &m2, kv, 0);
            for (int p = 0; p <= steps; p++)
                tern_forward(&m2, &s, toks[(size_t)p * nl + i], p);
            ok &= memcmp(s.logits, L[i]->logits, sizeof(float) * (size_t)m.cfg.vocab) == 0;
            tern_state_free(&s);
        }
        tern_free(&m2);
        printf("check: %d lanes bit-exact with single runs: %s\n",
               nl, ok ? "OK" : "FAIL");
    }
    printf("RESULT lanes=%d steps=%d mode=%s wall_ms=%.3f wait_ms=%.3f "
           "bytes_pass=%.0f macs_lane=%.0f wait_psram_ms=%.3f wait_flash_ms=%.3f "
           "wait_sd_ms=%.3f mismatch=%d slots=%d,%d,%d\n",
           nl, steps, io ? "emulated" : "sync", 1e3 * wall_med, 1e3 * wait_med,
           bytes_pass, macs_lane, 1e3 * w[0] / steps, 1e3 * w[1] / steps,
           1e3 * w[2] / steps, mismatch,
           io ? tern_io_slots(io, 0) : 0, io ? tern_io_slots(io, 1) : 0,
           io ? tern_io_slots(io, 2) : 0);
    if (io)
        tern_io_stop(io);
    for (int i = 0; i < nl; i++)
        tern_state_free(&st[i]);
    free(st); free(toks); free(tw); free(tws);
    tern_free(&m);
    return ok && !mismatch ? 0 : 1;
}
