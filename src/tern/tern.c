/* BitNet b1.58 forward pass, packed ternary GEMV and placement-driven weight reads. */
#include "tern.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int8_t LUT2[256][4];
static int8_t LUT3[256][8];
static int lut_ready = 0;

void tern_lut_init(void)
{
    if (lut_ready)
        return;
    memset(LUT3, 0, sizeof LUT3);
    for (int b = 0; b < 256; b++) {
        for (int k = 0; k < 4; k++)
            LUT2[b][k] = (int8_t)(((b >> (2 * k)) & 3) - 1);
        int v = b;
        for (int k = 0; k < 5; k++) {
            LUT3[b][k] = (int8_t)(v % 3 - 1);
            v /= 3;
        }
    }
    lut_ready = 1;
}

int tern_row_bytes(int fmt, int n)
{
    return fmt == TERN_FMT_T2 ? (n + 3) / 4 : (n + 4) / 5;
}

int tern_tile_bytes(int n)
{
    return ((n + 4) / 5) * 5 + 8;
}

void tern_pack_row(int fmt, const int8_t *w, int n, uint8_t *out)
{
    int nb = tern_row_bytes(fmt, n);
    for (int b = 0; b < nb; b++) {
        unsigned v = 0;
        if (fmt == TERN_FMT_T2) {
            for (int k = 0; k < 4; k++) {
                int i = b * 4 + k;
                unsigned d = (unsigned)((i < n ? w[i] : 0) + 1);
                v |= d << (2 * k);
            }
        } else {
            unsigned mul = 1;
            for (int k = 0; k < 5; k++) {
                int i = b * 5 + k;
                unsigned d = (unsigned)((i < n ? w[i] : 0) + 1);
                v += d * mul;
                mul *= 3;
            }
        }
        out[b] = (uint8_t)v;
    }
}

void tern_decode_row(int fmt, const uint8_t *p, int n, int8_t *tile)
{
    if (fmt == TERN_FMT_T2) {
        const int nb = (n + 3) / 4;
        for (int b = 0; b < nb; b++)
            memcpy(tile + 4 * b, LUT2[p[b]], 4);
    } else {
        const int nb = (n + 4) / 5;
        for (int b = 0; b < nb; b++)
            memcpy(tile + 5 * b, LUT3[p[b]], 8);
    }
}

int32_t tern_dot_i8(const int8_t *a, const int8_t *b, int n)
{
    int32_t s = 0;
    for (int i = 0; i < n; i++)
        s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

void tern_gemv_rows(int fmt, const uint8_t *data, int rows, int cols,
                    int row_bytes, const int8_t *xq, int8_t *tile, int32_t *acc)
{
    for (int r = 0; r < rows; r++) {
        tern_decode_row(fmt, data + (size_t)r * row_bytes, cols, tile);
        acc[r] = tern_dot_i8(xq, tile, cols);
    }
}

void tern_gemv_ref(int fmt, const uint8_t *data, int rows, int cols,
                   int row_bytes, const int8_t *xq, int32_t *acc)
{
    const int per = fmt == TERN_FMT_T2 ? 4 : 5;
    const int full = cols / per, tail = cols - full * per;
    for (int r = 0; r < rows; r++) {
        const uint8_t *p = data + (size_t)r * row_bytes;
        const int8_t *x = xq;
        int32_t a = 0;
        for (int b = 0; b < full; b++, x += per) {
            const int8_t *t = fmt == TERN_FMT_T2 ? LUT2[p[b]] : LUT3[p[b]];
            for (int k = 0; k < per; k++)
                a += t[k] * x[k];
        }
        if (tail) {
            const int8_t *t = fmt == TERN_FMT_T2 ? LUT2[p[full]] : LUT3[p[full]];
            for (int k = 0; k < tail; k++)
                a += t[k] * x[k];
        }
        acc[r] = a;
    }
}

float tern_quant_act(const float *x, int n, int8_t *xq)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > m)
            m = a;
    }
    const float s = 127.0f / (m > 1e-5f ? m : 1e-5f);
    for (int i = 0; i < n; i++) {
        float q = rintf(x[i] * s);
        if (q > 127.0f)
            q = 127.0f;
        if (q < -128.0f)
            q = -128.0f;
        xq[i] = (int8_t)q;
    }
    return s;
}

const char *tern_tier_name(int tier)
{
    static const char *names[TERN_NTIERS] = {"psram", "flash", "sd"};
    return tier >= 0 && tier < TERN_NTIERS ? names[tier] : "?";
}

static const uint8_t *seg_rows(const TernModel *m, TernState *s,
                               const TernMat *M, const TernSeg *g, int r,
                               int max_rows, int *n)
{
    const size_t rb = (size_t)M->row_bytes;
    int k = g->r1 - r;
    if (k > max_rows)
        k = max_rows;
    const uint8_t *p;
    if (g->data) {
        p = g->data + (size_t)(r - g->r0) * rb;
    } else {
        const int cap = (int)(s->sbuf_bytes / rb);
        if (k > cap)
            k = cap;
        const long off = g->off + (long)((size_t)(r - g->r0) * rb);
        if (!m->fp || fseek(m->fp, off, SEEK_SET) != 0
            || fread(s->sbuf, rb, (size_t)k, m->fp) != (size_t)k) {
            memset(s->sbuf, 0, rb * (size_t)k);
            s->io_errors++;
        }
        p = s->sbuf;
    }
    s->tier_bytes[g->tier] += (uint64_t)k * rb;
    s->tier_reads[g->tier]++;
    *n = k;
    return p;
}

static const uint8_t *mat_row(const TernModel *m, TernState *s,
                              const TernMat *M, int r)
{
    const TernSeg *g = M->seg;
    while (r >= g->r1)
        g++;
    int n;
    return seg_rows(m, s, M, g, r, 1, &n);
}

static int next_block(const TernMat *M, const int *cur, int per, int *seg, int *k)
{
    int best = -1;
    double bp = 2.0;
    for (int i = 0; i < M->nseg; i++) {
        const TernSeg *g = &M->seg[i];
        if (cur[i] >= g->r1)
            continue;
        const double pr = (double)(cur[i] - g->r0) / (double)(g->r1 - g->r0);
        if (pr < bp) {
            bp = pr;
            best = i;
        }
    }
    if (best < 0)
        return 0;
    *seg = best;
    *k = M->seg[best].r1 - cur[best];
    if (*k > per)
        *k = per;
    return 1;
}

static int rows_per_chunk(const TernMat *M, size_t chunk_bytes)
{
    int per = (int)(chunk_bytes / (size_t)M->row_bytes);
    return per < 1 ? 1 : per;
}

typedef void (*RowFn)(void *ctx, const uint8_t *p, int r, int n);

static void for_rows(const TernModel *m, TernState *s, const TernMat *M,
                     RowFn fn, void *ctx)
{
    int cur[TERN_NTIERS], seg, k;
    const int per = rows_per_chunk(M, s->chunk_bytes);
    for (int i = 0; i < M->nseg; i++)
        cur[i] = M->seg[i].r0;
    while (next_block(M, cur, per, &seg, &k)) {
        const TernSeg *g = &M->seg[seg];
        int n;
        const uint8_t *p;
        if (s->io) {
            p = s->io_fetch(s->io, M, seg, cur[seg], k);
            n = k;
            s->tier_bytes[g->tier] += (uint64_t)n * (uint64_t)M->row_bytes;
            s->tier_reads[g->tier]++;
        } else {
            p = seg_rows(m, s, M, g, cur[seg], k, &n);
        }
        fn(ctx, p, cur[seg], n);
        if (s->io)
            s->io_release(s->io, g->tier);
        cur[seg] += n;
    }
}

static void plan_mat(const TernMat *M, size_t chunk_bytes, TernPlanFn cb, void *ctx)
{
    int cur[TERN_NTIERS], seg, k;
    const int per = rows_per_chunk(M, chunk_bytes);
    for (int i = 0; i < M->nseg; i++)
        cur[i] = M->seg[i].r0;
    while (next_block(M, cur, per, &seg, &k)) {
        cb(ctx, M, seg, cur[seg], k);
        cur[seg] += k;
    }
}

void tern_plan(const TernModel *m, size_t chunk_bytes, TernPlanFn cb, void *ctx)
{
    for (int l = 0; l < m->cfg.n_layers; l++) {
        const TernLayer *L = &m->layers[l];
        const TernMat *seq[TERN_LINEARS] = {&L->q, &L->k, &L->v, &L->o,
                                            &L->gate, &L->up, &L->down};
        for (int j = 0; j < TERN_LINEARS; j++)
            plan_mat(seq[j], chunk_bytes, cb, ctx);
    }
    plan_mat(&m->head, chunk_bytes, cb, ctx);
}

static void trace(TernState *s, int pos, char kind, int unit, int r0, int r1,
                  uint64_t bytes, uint64_t macs)
{
    if (s->trace)
        fprintf(s->trace, "%d,%c,%d,%d,%d,%llu,%llu\n", pos, kind, unit, r0, r1,
                (unsigned long long)bytes, (unsigned long long)macs);
}

static void rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    ss = ss / (float)n;
    const float r = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; i++)
        y[i] = w[i] * (x[i] * r);
}

static void rope(float *v, int n_heads, int hd, int pos, float theta)
{
    const int half = hd / 2;
    for (int i = 0; i < half; i++) {
        const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
        const float ang = (float)pos * inv;
        const float c = cosf(ang), s = sinf(ang);
        for (int h = 0; h < n_heads; h++) {
            float *x = v + (size_t)h * hd;
            const float x1 = x[i], x2 = x[i + half];
            x[i] = x1 * c - x2 * s;
            x[i + half] = x2 * c + x1 * s;
        }
    }
}

typedef struct {
    const TernMat *M;
    TernState *const *L;
    int n, fmt;
} GemvCtx;

static void gemv_block(void *ctx, const uint8_t *p, int r, int nr)
{
    GemvCtx *g = (GemvCtx *)ctx;
    const int cols = g->M->cols;
    int8_t *tile = g->L[0]->tile;
    for (int j = 0; j < nr; j++) {
        tern_decode_row(g->fmt, p + (size_t)j * g->M->row_bytes, cols, tile);
        for (int i = 0; i < g->n; i++)
            g->L[i]->acc[r + j] = tern_dot_i8(g->L[i]->xq, tile, cols);
    }
}

static void bitlinear(const TernModel *m, TernState *const *L, int n,
                      const TernMat *M, const float *sc, size_t out_off, int pos)
{
    GemvCtx g;
    g.M = M;
    g.L = L;
    g.n = n;
    g.fmt = m->cfg.fmt;
    for_rows(m, L[0], M, gemv_block, &g);
    for (int i = 0; i < n; i++) {
        float *y = *(float **)(void *)((char *)L[i] + out_off);
        for (int j = 0; j < M->rows; j++)
            y[j] = ((float)L[i]->acc[j] / sc[i]) * M->scale;
    }
    const uint64_t bytes = (uint64_t)M->rows * (uint64_t)M->row_bytes;
    const uint64_t macs = (uint64_t)M->rows * (uint64_t)M->cols * (uint64_t)n;
    L[0]->bytes_ternary += bytes;
    L[0]->macs += macs;
    trace(L[0], pos, 'W', M->unit, 0, M->rows, bytes, macs);
}

static void kv_quant(const float *v, int nkv, int hd, int8_t *q, float *sc)
{
    for (int h = 0; h < nkv; h++) {
        const float *x = v + (size_t)h * hd;
        float mx = 0.0f;
        for (int i = 0; i < hd; i++)
            if (fabsf(x[i]) > mx)
                mx = fabsf(x[i]);
        const float e = mx > 0.0f ? mx / 127.0f : 1.0f;
        for (int i = 0; i < hd; i++) {
            float r = rintf(x[i] / e);
            if (r > 127.0f)
                r = 127.0f;
            if (r < -127.0f)
                r = -127.0f;
            q[(size_t)h * hd + i] = (int8_t)r;
        }
        sc[h] = e;
    }
}

static uint64_t attend(const TernModel *m, TernState *s, int l, int pos)
{
    const TernConfig *c = &m->cfg;
    const int hd = c->head_dim, nh = c->n_heads, nkv = c->n_kv;
    const int kvdim = nkv * hd, group = nh / nkv;
    const float scale = (float)(1.0 / sqrt((double)hd));
    const int kv8 = s->kv_fmt == TERN_KV_I8;
    const size_t base = (size_t)l * c->max_seq;

    rope(s->q, nh, hd, pos, c->rope_theta);
    rope(s->k, nkv, hd, pos, c->rope_theta);
    if (kv8) {
        int8_t *kc = (int8_t *)s->kcache, *vc = (int8_t *)s->vcache;
        kv_quant(s->k, nkv, hd, kc + (base + pos) * kvdim,
                 s->kscale + (base + pos) * nkv);
        kv_quant(s->v, nkv, hd, vc + (base + pos) * kvdim,
                 s->vscale + (base + pos) * nkv);
    } else {
        float *kc = (float *)s->kcache, *vc = (float *)s->vcache;
        memcpy(kc + (base + pos) * kvdim, s->k, (size_t)kvdim * sizeof(float));
        memcpy(vc + (base + pos) * kvdim, s->v, (size_t)kvdim * sizeof(float));
    }
    for (int h = 0; h < nh; h++) {
        const int kh = h / group;
        const float *qh = s->q + (size_t)h * hd;
        float mx = -INFINITY;
        for (int t = 0; t <= pos; t++) {
            const size_t row = (base + t) * kvdim + (size_t)kh * hd;
            float d = 0.0f;
            if (kv8) {
                const int8_t *kt = (const int8_t *)s->kcache + row;
                const float e = s->kscale[(base + t) * nkv + kh];
                for (int i = 0; i < hd; i++)
                    d += qh[i] * ((float)kt[i] * e);
            } else {
                const float *kt = (const float *)s->kcache + row;
                for (int i = 0; i < hd; i++)
                    d += qh[i] * kt[i];
            }
            d = d * scale;
            s->scores[t] = d;
            if (d > mx)
                mx = d;
        }
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            s->scores[t] = expf(s->scores[t] - mx);
            sum += s->scores[t];
        }
        float *out = s->att + (size_t)h * hd;
        for (int i = 0; i < hd; i++)
            out[i] = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float pr = s->scores[t] / sum;
            const size_t row = (base + t) * kvdim + (size_t)kh * hd;
            if (kv8) {
                const int8_t *vt = (const int8_t *)s->vcache + row;
                const float e = s->vscale[(base + t) * nkv + kh];
                for (int i = 0; i < hd; i++)
                    out[i] += pr * ((float)vt[i] * e);
            } else {
                const float *vt = (const float *)s->vcache + row;
                for (int i = 0; i < hd; i++)
                    out[i] += pr * vt[i];
            }
        }
    }
    const uint64_t per_pos = kv8 ? (uint64_t)(kvdim + nkv * (int)sizeof(float))
                                 : (uint64_t)kvdim * sizeof(float);
    return (uint64_t)(pos + 1) * per_pos * 2u;
}

typedef struct {
    TernState *const *L;
    int n, dim, hrb;
    const float *sa;
} HeadCtx;

static void head_block_f32(void *ctx, const uint8_t *p, int r, int nr)
{
    HeadCtx *h = (HeadCtx *)ctx;
    for (int j = 0; j < nr; j++) {
        const float *e = (const float *)(const void *)(p + (size_t)j * h->hrb);
        for (int i = 0; i < h->n; i++) {
            const float *xn = h->L[i]->xn;
            float d = 0.0f;
            for (int k = 0; k < h->dim; k++)
                d += e[k] * xn[k];
            h->L[i]->logits[r + j] = d;
        }
    }
}

static void head_block_i8(void *ctx, const uint8_t *p, int r, int nr)
{
    HeadCtx *h = (HeadCtx *)ctx;
    for (int j = 0; j < nr; j++) {
        const uint8_t *row = p + (size_t)j * h->hrb;
        float e;
        memcpy(&e, row, sizeof e);
        for (int i = 0; i < h->n; i++) {
            const int32_t acc = tern_dot_i8(h->L[i]->xq, (const int8_t *)(row + 4),
                                            h->dim);
            h->L[i]->logits[r + j] = ((float)acc / h->sa[i]) * e;
        }
    }
}

#define OFF(field) offsetof(TernState, field)

void tern_forward_batch(const TernModel *m, TernState *const *L, int n,
                        const int *tokens, int pos)
{
    const TernConfig *c = &m->cfg;
    const int dim = c->dim, qdim = c->n_heads * c->head_dim;
    const int hrb = m->head.row_bytes;
    TernState *s0 = L[0];
    float sc[TERN_MAX_LANES];

    for (int i = 0; i < n; i++) {
        TernState *s = L[i];
        const uint8_t *p = mat_row(m, s0, &m->head, tokens[i]);
        if (c->head_fmt == TERN_HEAD_F32) {
            memcpy(s->x, p, (size_t)dim * sizeof(float));
        } else {
            float e;
            memcpy(&e, p, sizeof e);
            const int8_t *q = (const int8_t *)(p + 4);
            for (int k = 0; k < dim; k++)
                s->x[k] = (float)q[k] * e;
        }
        s0->bytes_emb += (uint64_t)hrb;
        trace(s0, pos, 'E', 0, tokens[i], tokens[i] + 1, (uint64_t)hrb, 0);
    }

    for (int l = 0; l < c->n_layers; l++) {
        const TernLayer *Ly = &m->layers[l];
        const uint64_t aux = (uint64_t)(2 * dim + qdim + c->inter) * sizeof(float);
        s0->bytes_aux += aux;
        trace(s0, pos, 'A', l, 0, 0, aux, 0);

        for (int i = 0; i < n; i++) {
            rmsnorm(L[i]->xn, L[i]->x, Ly->attn_norm, dim, c->eps);
            sc[i] = tern_quant_act(L[i]->xn, dim, L[i]->xq);
        }
        bitlinear(m, L, n, &Ly->q, sc, OFF(q), pos);
        bitlinear(m, L, n, &Ly->k, sc, OFF(k), pos);
        bitlinear(m, L, n, &Ly->v, sc, OFF(v), pos);
        {
            uint64_t kvb = 0, kvm = 0;
            for (int i = 0; i < n; i++)
                kvb += attend(m, L[i], l, pos);
            kvm = 2u * (uint64_t)c->n_heads * (uint64_t)(pos + 1)
                * (uint64_t)c->head_dim * (uint64_t)n;
            s0->bytes_kv += kvb;
            trace(s0, pos, 'K', l, 0, pos + 1, kvb, kvm);
        }
        for (int i = 0; i < n; i++) {
            rmsnorm(L[i]->att, L[i]->att, Ly->attn_sub_norm, qdim, c->eps);
            sc[i] = tern_quant_act(L[i]->att, qdim, L[i]->xq);
        }
        bitlinear(m, L, n, &Ly->o, sc, OFF(o), pos);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < dim; k++)
                L[i]->x[k] += L[i]->o[k];

        for (int i = 0; i < n; i++) {
            rmsnorm(L[i]->xn, L[i]->x, Ly->ffn_norm, dim, c->eps);
            sc[i] = tern_quant_act(L[i]->xn, dim, L[i]->xq);
        }
        bitlinear(m, L, n, &Ly->gate, sc, OFF(g), pos);
        bitlinear(m, L, n, &Ly->up, sc, OFF(u), pos);
        for (int i = 0; i < n; i++) {
            TernState *s = L[i];
            for (int k = 0; k < c->inter; k++) {
                const float r = s->g[k] > 0.0f ? s->g[k] : 0.0f;
                s->h[k] = (r * r) * s->u[k];
            }
            rmsnorm(s->h, s->h, Ly->ffn_sub_norm, c->inter, c->eps);
            sc[i] = tern_quant_act(s->h, c->inter, s->xq);
        }
        bitlinear(m, L, n, &Ly->down, sc, OFF(o), pos);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < dim; k++)
                L[i]->x[k] += L[i]->o[k];
    }

    s0->bytes_aux += (uint64_t)dim * sizeof(float);
    trace(s0, pos, 'A', c->n_layers, 0, 0, (uint64_t)dim * sizeof(float), 0);
    for (int i = 0; i < n; i++) {
        rmsnorm(L[i]->xn, L[i]->x, m->final_norm, dim, c->eps);
        sc[i] = c->head_fmt == TERN_HEAD_I8 ? tern_quant_act(L[i]->xn, dim, L[i]->xq)
                                           : 0.0f;
    }
    {
        HeadCtx hc;
        hc.L = L;
        hc.n = n;
        hc.dim = dim;
        hc.hrb = hrb;
        hc.sa = sc;
        for_rows(m, s0, &m->head,
                 c->head_fmt == TERN_HEAD_F32 ? head_block_f32 : head_block_i8, &hc);
    }
    {
        const uint64_t hb = (uint64_t)c->vocab * (uint64_t)hrb;
        const uint64_t hm = (uint64_t)c->vocab * (uint64_t)dim * (uint64_t)n;
        s0->bytes_head += hb;
        s0->macs += hm;
        trace(s0, pos, 'H', 0, 0, c->vocab, hb, hm);
    }
}

const float *tern_forward(const TernModel *m, TernState *s, int token, int pos)
{
    TernState *L[1];
    L[0] = s;
    tern_forward_batch(m, L, 1, &token, pos);
    return s->logits;
}

static uint32_t rd32(const uint8_t *b)
{
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16
         | (uint32_t)b[3] << 24;
}

static long align16(long x)
{
    return (x + 15) & ~15L;
}

static int read_at(FILE *f, long off, void *dst, size_t n)
{
    if (n == 0)
        return 0;
    return fseek(f, off, SEEK_SET) == 0 && fread(dst, 1, n, f) == n ? 0 : -1;
}

static TernMat *unit_mat(TernModel *m, int u)
{
    if (u == 0)
        return &m->head;
    TernLayer *L = &m->layers[(u - 1) / TERN_LINEARS];
    switch ((u - 1) % TERN_LINEARS) {
    case 0: return &L->q;
    case 1: return &L->k;
    case 2: return &L->v;
    case 3: return &L->o;
    case 4: return &L->gate;
    case 5: return &L->up;
    default: return &L->down;
    }
}

typedef struct {
    int n;
    int tier[TERN_NTIERS], r0[TERN_NTIERS], r1[TERN_NTIERS];
} Plan;

static int parse_tier(const char *s)
{
    for (int t = 0; t < TERN_NTIERS; t++)
        if (strcmp(s, tern_tier_name(t)) == 0)
            return t;
    return -1;
}

static int read_placement(const char *path, Plan *plan, int nunits,
                          char *err, size_t errlen)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(err, errlen, "cannot open placement %s", path);
        return -1;
    }
    char line[256];
    int ln = 0;
    while (fgets(line, sizeof line, f)) {
        ln++;
        char *p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
            continue;
        int u, r0, r1;
        char tname[16];
        if (sscanf(p, "%d %15s %d %d", &u, tname, &r0, &r1) != 4) {
            snprintf(err, errlen, "placement: malformed line %d", ln);
            fclose(f);
            return -1;
        }
        const int t = parse_tier(tname);
        if (u < 0 || u >= nunits || t < 0 || r0 < 0 || r1 <= r0
            || plan[u].n >= TERN_NTIERS) {
            snprintf(err, errlen, "placement: invalid line %d", ln);
            fclose(f);
            return -1;
        }
        Plan *P = &plan[u];
        P->tier[P->n] = t;
        P->r0[P->n] = r0;
        P->r1[P->n] = r1;
        P->n++;
    }
    fclose(f);
    return 0;
}

int tern_open(TernModel *m, const char *path, const char *place_path,
              char *err, size_t errlen)
{
    memset(m, 0, sizeof(*m));
    tern_lut_init();

    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, errlen, "cannot open %s", path);
        return -1;
    }
    uint8_t hdr[64];
    if (fread(hdr, 1, 64, f) != 64 || memcmp(hdr, TERN_MAGIC, 4) != 0) {
        fclose(f);
        snprintf(err, errlen, "bad header");
        return -1;
    }
    if (rd32(hdr + 4) != TERN_VERSION) {
        fclose(f);
        snprintf(err, errlen, "unsupported version %u", rd32(hdr + 4));
        return -1;
    }
    TernConfig *c = &m->cfg;
    c->fmt = (int)rd32(hdr + 8);
    c->dim = (int)rd32(hdr + 12);
    c->n_layers = (int)rd32(hdr + 16);
    c->n_heads = (int)rd32(hdr + 20);
    c->n_kv = (int)rd32(hdr + 24);
    c->head_dim = (int)rd32(hdr + 28);
    c->inter = (int)rd32(hdr + 32);
    c->vocab = (int)rd32(hdr + 36);
    c->max_seq = (int)rd32(hdr + 40);
    memcpy(&c->rope_theta, hdr + 44, 4);
    memcpy(&c->eps, hdr + 48, 4);
    c->head_fmt = (int)rd32(hdr + 52);
    if ((c->fmt != TERN_FMT_T2 && c->fmt != TERN_FMT_T3)
        || (c->head_fmt != TERN_HEAD_F32 && c->head_fmt != TERN_HEAD_I8)
        || c->n_kv <= 0 || c->n_heads % c->n_kv != 0 || c->head_dim % 2 != 0
        || c->dim <= 0 || c->n_layers <= 0 || c->vocab <= 0 || c->max_seq <= 0) {
        fclose(f);
        snprintf(err, errlen, "invalid configuration");
        return -1;
    }
    const int dim = c->dim, qdim = c->n_heads * c->head_dim;
    const int kvdim = c->n_kv * c->head_dim, L = c->n_layers;
    const int nunits = 1 + TERN_LINEARS * L;

    m->layers = (TernLayer *)calloc((size_t)L, sizeof(TernLayer));
    Plan *plan = (Plan *)calloc((size_t)nunits, sizeof(Plan));
    if (!m->layers || !plan) {
        free(plan);
        fclose(f);
        tern_free(m);
        snprintf(err, errlen, "out of memory");
        return -1;
    }

    long off = 64;
    TernMat *H = &m->head;
    H->unit = 0;
    H->rows = c->vocab;
    H->cols = dim;
    H->row_bytes = c->head_fmt == TERN_HEAD_F32 ? 4 * dim : dim + 4;
    H->seg[0].off = off;
    off = align16(off + (long)c->vocab * H->row_bytes);
    const long final_norm_off = off;
    off = align16(off + 4L * dim);
    const int lin_rows[TERN_LINEARS] = {qdim, kvdim, kvdim, dim, c->inter, c->inter, dim};
    const int lin_cols[TERN_LINEARS] = {dim, dim, dim, qdim, dim, dim, c->inter};
    const int norm_len[4] = {dim, qdim, dim, c->inter};
    long *norm_off = (long *)malloc(sizeof(long) * 4 * (size_t)L);
    long *scale_off = (long *)malloc(sizeof(long) * TERN_LINEARS * (size_t)L);
    if (!norm_off || !scale_off) {
        free(norm_off); free(scale_off); free(plan);
        fclose(f);
        tern_free(m);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    for (int l = 0; l < L; l++) {
        for (int k = 0; k < 4; k++) {
            norm_off[4 * l + k] = off;
            off = align16(off + 4L * norm_len[k]);
        }
        for (int j = 0; j < TERN_LINEARS; j++) {
            TernMat *M = unit_mat(m, 1 + TERN_LINEARS * l + j);
            M->unit = 1 + TERN_LINEARS * l + j;
            M->rows = lin_rows[j];
            M->cols = lin_cols[j];
            M->row_bytes = tern_row_bytes(c->fmt, M->cols);
            scale_off[TERN_LINEARS * l + j] = off;
            off = align16(off + 4);
            M->seg[0].off = off;
            off = align16(off + (long)M->rows * M->row_bytes);
        }
    }
    long fsize = -1;
    if (fseek(f, 0, SEEK_END) == 0)
        fsize = ftell(f);
    if (fsize != off) {
        free(norm_off); free(scale_off); free(plan);
        fclose(f);
        tern_free(m);
        snprintf(err, errlen, "file size %ld, expected %ld", fsize, off);
        return -1;
    }

    size_t aux = (size_t)dim;
    for (int k = 0; k < 4; k++)
        aux += (size_t)L * (size_t)norm_len[k];
    m->aux_bytes = aux * sizeof(float);
    m->aux = (uint8_t *)malloc(m->aux_bytes);
    int bad = m->aux == NULL;
    float *ap = (float *)(void *)m->aux;
    if (!bad) {
        bad |= read_at(f, final_norm_off, ap, 4u * (size_t)dim);
        m->final_norm = ap;
        ap += dim;
    }
    for (int l = 0; l < L && !bad; l++) {
        const float **dst[4] = {&m->layers[l].attn_norm, &m->layers[l].attn_sub_norm,
                                &m->layers[l].ffn_norm, &m->layers[l].ffn_sub_norm};
        for (int k = 0; k < 4; k++) {
            bad |= read_at(f, norm_off[4 * l + k], ap, 4u * (size_t)norm_len[k]);
            *dst[k] = ap;
            ap += norm_len[k];
        }
        for (int j = 0; j < TERN_LINEARS; j++) {
            uint8_t sb[4];
            bad |= read_at(f, scale_off[TERN_LINEARS * l + j], sb, 4);
            memcpy(&unit_mat(m, 1 + TERN_LINEARS * l + j)->scale, sb, 4);
        }
    }
    free(norm_off);
    free(scale_off);
    if (bad) {
        free(plan);
        fclose(f);
        tern_free(m);
        snprintf(err, errlen, "failed to read norms/scales");
        return -1;
    }

    if (place_path && read_placement(place_path, plan, nunits, err, errlen) != 0) {
        free(plan);
        fclose(f);
        tern_free(m);
        return -1;
    }
    for (int u = 0; u < nunits; u++) {
        TernMat *M = unit_mat(m, u);
        Plan *P = &plan[u];
        if (P->n == 0) {
            P->n = 1;
            P->tier[0] = TERN_TIER_PSRAM;
            P->r0[0] = 0;
            P->r1[0] = M->rows;
        }
        for (int i = 1; i < P->n; i++)
            for (int j = i; j > 0 && P->r0[j] < P->r0[j - 1]; j--) {
                int t;
                t = P->tier[j]; P->tier[j] = P->tier[j - 1]; P->tier[j - 1] = t;
                t = P->r0[j]; P->r0[j] = P->r0[j - 1]; P->r0[j - 1] = t;
                t = P->r1[j]; P->r1[j] = P->r1[j - 1]; P->r1[j - 1] = t;
            }
        int expect = 0;
        for (int i = 0; i < P->n; i++) {
            if (P->r0[i] != expect)
                break;
            expect = P->r1[i];
        }
        if (expect != M->rows) {
            free(plan);
            fclose(f);
            tern_free(m);
            snprintf(err, errlen, "placement: unit %d does not cover its %d rows "
                     "with contiguous segments", u, M->rows);
            return -1;
        }
        const long base = M->seg[0].off;
        M->nseg = P->n;
        for (int i = 0; i < P->n; i++) {
            TernSeg *g = &M->seg[i];
            g->tier = P->tier[i];
            g->r0 = P->r0[i];
            g->r1 = P->r1[i];
            g->data = NULL;
            g->off = base + (long)P->r0[i] * M->row_bytes;
            m->placed[g->tier] += (size_t)(g->r1 - g->r0) * (size_t)M->row_bytes;
            if (g->tier == TERN_TIER_SD && (size_t)M->row_bytes > m->max_stream_row)
                m->max_stream_row = (size_t)M->row_bytes;
        }
    }
    free(plan);

    size_t cur[TERN_NTIERS] = {0, 0, 0}, need[TERN_NTIERS] = {0, 0, 0};
    for (int u = 0; u < nunits; u++) {
        const TernMat *M = unit_mat(m, u);
        for (int i = 0; i < M->nseg; i++)
            need[M->seg[i].tier] += ((size_t)(M->seg[i].r1 - M->seg[i].r0)
                                     * (size_t)M->row_bytes + 15u) & ~(size_t)15u;
    }
    for (int t = 0; t < TERN_NTIERS; t++)
        if (t != TERN_TIER_SD && need[t] > 0) {
            m->arena[t] = (uint8_t *)malloc(need[t]);
            if (!m->arena[t]) {
                fclose(f);
                tern_free(m);
                snprintf(err, errlen, "out of memory for %zu bytes in %s",
                         need[t], tern_tier_name(t));
                return -1;
            }
        }
    for (int u = 0; u < nunits && !bad; u++) {
        TernMat *M = unit_mat(m, u);
        for (int i = 0; i < M->nseg; i++) {
            TernSeg *g = &M->seg[i];
            if (g->tier == TERN_TIER_SD)
                continue;
            const size_t nb = (size_t)(g->r1 - g->r0) * (size_t)M->row_bytes;
            uint8_t *dst = m->arena[g->tier] + cur[g->tier];
            bad |= read_at(f, g->off, dst, nb);
            g->data = dst;
            cur[g->tier] += (nb + 15u) & ~(size_t)15u;
        }
    }
    if (bad) {
        fclose(f);
        tern_free(m);
        snprintf(err, errlen, "failed to read weights");
        return -1;
    }
    if (m->placed[TERN_TIER_SD] > 0)
        m->fp = f;
    else
        fclose(f);
    return 0;
}

int tern_load(TernModel *m, const char *path, char *err, size_t errlen)
{
    return tern_open(m, path, NULL, err, errlen);
}

void tern_free(TernModel *m)
{
    for (int t = 0; t < TERN_NTIERS; t++)
        free(m->arena[t]);
    free(m->layers);
    free(m->aux);
    if (m->fp)
        fclose(m->fp);
    memset(m, 0, sizeof(*m));
}

int tern_state_init(TernState *s, const TernModel *m, int kv_fmt,
                    size_t stream_buf)
{
    const TernConfig *c = &m->cfg;
    memset(s, 0, sizeof(*s));
    s->kv_fmt = kv_fmt;
    s->chunk_bytes = TERN_CHUNK_DEFAULT;
    const int qdim = c->n_heads * c->head_dim, kvdim = c->n_kv * c->head_dim;
    int big = c->dim;
    if (qdim > big)
        big = qdim;
    if (c->inter > big)
        big = c->inter;
    s->x = calloc((size_t)c->dim, sizeof(float));
    s->xn = calloc((size_t)c->dim, sizeof(float));
    s->q = calloc((size_t)qdim, sizeof(float));
    s->k = calloc((size_t)kvdim, sizeof(float));
    s->v = calloc((size_t)kvdim, sizeof(float));
    s->att = calloc((size_t)qdim, sizeof(float));
    s->o = calloc((size_t)c->dim, sizeof(float));
    s->g = calloc((size_t)c->inter, sizeof(float));
    s->u = calloc((size_t)c->inter, sizeof(float));
    s->h = calloc((size_t)c->inter, sizeof(float));
    s->scores = calloc((size_t)c->max_seq, sizeof(float));
    s->logits = calloc((size_t)c->vocab, sizeof(float));
    s->xq = calloc((size_t)big, sizeof(int8_t));
    s->tile = calloc((size_t)tern_tile_bytes(big), sizeof(int8_t));
    s->acc = calloc((size_t)big, sizeof(int32_t));
    const size_t kv = (size_t)c->n_layers * c->max_seq * kvdim;
    const size_t ksz = kv_fmt == TERN_KV_I8 ? 1 : sizeof(float);
    s->kcache = calloc(kv, ksz);
    s->vcache = calloc(kv, ksz);
    int bad = !s->x || !s->xn || !s->q || !s->k || !s->v || !s->att || !s->o
           || !s->g || !s->u || !s->h || !s->scores || !s->logits || !s->xq
           || !s->tile || !s->acc || !s->kcache || !s->vcache;
    if (kv_fmt == TERN_KV_I8) {
        const size_t ns = (size_t)c->n_layers * c->max_seq * c->n_kv;
        s->kscale = calloc(ns, sizeof(float));
        s->vscale = calloc(ns, sizeof(float));
        bad |= !s->kscale || !s->vscale;
    }
    if (m->placed[TERN_TIER_SD] > 0) {
        s->sbuf_bytes = stream_buf ? stream_buf : 65536u;
        if (s->sbuf_bytes < m->max_stream_row)
            s->sbuf_bytes = m->max_stream_row;
        if (s->sbuf_bytes < s->chunk_bytes)
            s->sbuf_bytes = s->chunk_bytes;
        s->sbuf = malloc(s->sbuf_bytes);
        bad |= !s->sbuf;
    }
    if (bad) {
        tern_state_free(s);
        return -1;
    }
    return 0;
}

int tern_set_chunk(TernState *s, size_t chunk_bytes)
{
    s->chunk_bytes = chunk_bytes;
    if (s->sbuf && s->sbuf_bytes < chunk_bytes) {
        uint8_t *b = (uint8_t *)realloc(s->sbuf, chunk_bytes);
        if (!b)
            return -1;
        s->sbuf = b;
        s->sbuf_bytes = chunk_bytes;
    }
    return 0;
}

void tern_state_free(TernState *s)
{
    free(s->x); free(s->xn); free(s->q); free(s->k); free(s->v); free(s->att);
    free(s->o); free(s->g); free(s->u); free(s->h); free(s->scores);
    free(s->logits); free(s->xq); free(s->tile); free(s->acc);
    free(s->kcache); free(s->vcache); free(s->kscale); free(s->vscale);
    free(s->sbuf);
    memset(s, 0, sizeof(*s));
}

void tern_state_reset_counters(TernState *s)
{
    s->bytes_ternary = s->bytes_head = s->bytes_emb = s->bytes_kv = 0;
    s->bytes_aux = s->macs = s->io_errors = 0;
    for (int t = 0; t < TERN_NTIERS; t++)
        s->tier_bytes[t] = s->tier_reads[t] = 0;
}
