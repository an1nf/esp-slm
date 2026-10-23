/* f16 conversion, PTQ1_0 decode, Q8 activations, ternary GEMV, norms, Hadamard. */
#include "q35_int.h"

uint16_t q35_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ex = (x >> 23) & 0xffu;
    uint32_t man = x & 0x7fffffu;
    if (ex == 0xffu)
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u | (man >> 13) : 0u));
    const int e = (int)ex - 127 + 15;
    if (e >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10)
            return (uint16_t)sign;
        man |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = man >> sh;
        const uint32_t rem = man & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u)))
            r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((uint32_t)e << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u)))
        r++;
    return (uint16_t)(sign | r);
}

float q35_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    uint32_t x;
    float f;
    if (ex == 0) {
        f = (float)man * 5.9604644775390625e-8f;
        return sign ? -f : f;
    }
    if (ex == 31)
        x = sign | 0x7f800000u | (man << 13);
    else
        x = sign | ((ex + 112u) << 23) | (man << 13);
    memcpy(&f, &x, 4);
    return f;
}

float rd_f16(const uint8_t *p)
{
    return q35_f16_to_f32((uint16_t)(p[0] | p[1] << 8));
}

static int8_t LUT5[256][5];
static int lut_ready;

static void lut_init(void)
{
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int b = 0; b < 256; b++)
        for (int n = 0; n < 5; n++) {
            const uint8_t v = (uint8_t)(b * pow3[n]);
            LUT5[b][n] = (int8_t)((((uint16_t)v * 3u) >> 8) - 1);
        }
    lut_ready = 1;
}

void q35_decode_ptq1(const uint8_t *b, int8_t *t)
{
    if (!lut_ready)
        lut_init();
    for (int n = 0; n < 5; n++)
        for (int m = 0; m < 16; m++)
            t[16 * n + m] = LUT5[b[m]][n];
    for (int n = 0; n < 5; n++)
        for (int m = 0; m < 8; m++)
            t[80 + 8 * n + m] = LUT5[b[16 + m]][n];
    for (int n = 0; n < 4; n++)
        for (int h = 0; h < 2; h++)
            t[120 + 2 * n + h] = LUT5[b[24 + h]][n];
}

void q35_quant_q8(const float *x, int n, int8_t *q, float *d)
{
    for (int b = 0; b < n / 32; b++) {
        const float *xb = x + 32 * b;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            const float a = fabsf(xb[j]);
            if (a > amax)
                amax = a;
        }
        d[b] = q35_f16_to_f32(q35_f32_to_f16(amax / 127.0f));
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (int j = 0; j < 32; j++)
            q[32 * b + j] = (int8_t)rintf(xb[j] * id);
    }
}

void rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    const float mean = ss / (float)n;
    const float sc = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < n; i++)
        y[i] = (x[i] * sc) * w[i];
}

void l2norm(float *x, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    const float sc = 1.0f / fmaxf(sqrtf(ss), eps);
    for (int i = 0; i < n; i++)
        x[i] = x[i] * sc;
}

static void count(Q35State *s, const Q35Mat *W, int T, int rows)
{
    const uint64_t bytes = (uint64_t)rows * W->row_bytes;
    const uint64_t macs = (uint64_t)rows * (uint64_t)W->cols * (uint64_t)T;
    s->bytes_kind[W->kind] += bytes;
    s->macs += macs;
    if (s->trace)
        fprintf(s->trace, "W %d %d %d %d %llu %llu %d\n", W->kind, W->layer,
                rows, W->cols, (unsigned long long)bytes,
                (unsigned long long)macs, T);
}

static int needs_q8(int type)
{
    return type == GGML_PTQ1_0 || type == GGML_Q8_0;
}

static void fwht_blocks(float *x, int n, int B)
{
    const float sc = 1.0f / sqrtf((float)B);
    for (int o = 0; o < n; o += B) {
        float *v = x + o;
        for (int len = 1; len < B; len <<= 1)
            for (int i = 0; i < B; i += 2 * len)
                for (int j = i; j < i + len; j++) {
                    const float a = v[j], b = v[j + len];
                    v[j] = a + b;
                    v[j + len] = a - b;
                }
        for (int i = 0; i < B; i++)
            v[i] = v[i] * sc;
    }
}

static void rot_apply(const Q35Rot *R, const float *x, float *y, int n)
{
    if (R->perm_rep > 1) {
        const int hd = R->perm_hd, nk = R->perm_nk, rep = R->perm_rep;
        for (int k = 0; k < nk; k++)
            for (int r = 0; r < rep; r++)
                memcpy(y + (size_t)hd * (r + rep * k), x + (size_t)hd * (k + nk * r),
                       (size_t)hd * sizeof(float));
    } else {
        memcpy(y, x, (size_t)n * sizeof(float));
    }
    if (R->signs)
        for (int i = 0; i < n; i++)
            y[i] = y[i] * R->signs[i];
    fwht_blocks(y, n, R->block);
}

void rot_inverse(const Q35Rot *R, float *x, int n)
{
    fwht_blocks(x, n, R->block);
    if (R->signs)
        for (int i = 0; i < n; i++)
            x[i] = x[i] * R->signs[i];
}

int rot_sig(const Q35Rot *R)
{
    return R->block ? 1 + R->block * 4 + (R->perm_rep > 1) * 2 + (R->signs != NULL) : 0;
}

static void act_q8(Q35State *s, const float *x, int xs, int cols, int T)
{
    for (int t = 0; t < T; t++)
        q35_quant_q8(x + (size_t)t * xs, cols, s->aq + (size_t)t * cols,
                     s->ad + (size_t)t * (cols / 32));
}

static void row_dot(const Q35Mat *W, const Q35State *s, int T, const float *x, int xs,
                    float *y, int ys, int r, int8_t *tile, float *td)
{
    const int cols = W->cols;
    const uint8_t *p = W->data + (size_t)r * W->row_bytes;
    if (W->type == GGML_PTQ1_0) {
        const int nb = cols / 128;
        for (int b = 0; b < nb; b++) {
            q35_decode_ptq1(p + 28 * b, tile + 128 * b);
            td[b] = rd_f16(p + 28 * b + 26);
        }
        for (int t = 0; t < T; t++) {
            const int8_t *q = s->aq + (size_t)t * cols;
            const float *dq = s->ad + (size_t)t * (cols / 32);
            float sumf = 0.0f;
            for (int b = 0; b < nb; b++) {
                float sumi = 0.0f;
                for (int k = 0; k < 4; k++) {
                    const int8_t *w = tile + 128 * b + 32 * k;
                    const int8_t *a = q + 128 * b + 32 * k;
                    int32_t si = 0;
                    for (int e = 0; e < 32; e++)
                        si += (int32_t)w[e] * (int32_t)a[e];
                    sumi += dq[4 * b + k] * (float)si;
                }
                sumf += td[b] * sumi;
            }
            y[(size_t)t * ys + r] = W->has_scale ? sumf * W->scale : sumf;
        }
    } else if (W->type == GGML_Q8_0) {
        const int nb = cols / 32;
        for (int t = 0; t < T; t++) {
            const int8_t *q = s->aq + (size_t)t * cols;
            const float *dq = s->ad + (size_t)t * nb;
            float sumf = 0.0f;
            for (int b = 0; b < nb; b++) {
                const int8_t *w = (const int8_t *)(p + 34 * b + 2);
                int32_t si = 0;
                for (int e = 0; e < 32; e++)
                    si += (int32_t)w[e] * (int32_t)q[32 * b + e];
                sumf += (float)si * (rd_f16(p + 34 * b) * dq[b]);
            }
            y[(size_t)t * ys + r] = W->has_scale ? sumf * W->scale : sumf;
        }
    } else if (W->type == GGML_F32) {
        const float *w = (const float *)(const void *)p;
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * xs;
            float sum = 0.0f;
            for (int i = 0; i < cols; i++)
                sum += w[i] * xt[i];
            y[(size_t)t * ys + r] = W->has_scale ? sum * W->scale : sum;
        }
    } else {
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * xs;
            float sum = 0.0f;
            for (int i = 0; i < cols; i++)
                sum += rd_f16(p + 2 * i) * xt[i];
            y[(size_t)t * ys + r] = W->has_scale ? sum * W->scale : sum;
        }
    }
}

/* y[T x rows] = W x for T tokens: one decode of each weight row, T int8 dots. */
void linear(const Q35Mat *W, Q35State *s, int T, const float *x, int xs,
                   int *qready, float *y, int ys)
{
    if (*qready == 0)
        *qready = ++s->act_gen;
    const int g = *qready, sig = rot_sig(&W->rot);
    if (sig) {
        if (s->hx_gen != g || s->hx_sig != sig) {
            for (int t = 0; t < T; t++)
                rot_apply(&W->rot, x + (size_t)t * xs, s->hx + (size_t)t * W->cols, W->cols);
            s->hx_gen = g;
            s->hx_sig = sig;
        }
        x = s->hx;
        xs = W->cols;
    }
    if (needs_q8(W->type) && (s->aq_gen != g || s->aq_sig != sig)) {
        act_q8(s, x, xs, W->cols, T);
        s->aq_gen = g;
        s->aq_sig = sig;
    }
    const int rows = W->rows;
#ifdef _OPENMP
#pragma omp parallel num_threads(s->nthreads)
    {
        const int id = omp_get_thread_num();
        int8_t *tile = s->tile + (size_t)id * s->tile_stride;
        float *td = s->td + (size_t)id * (s->tile_stride / 128 + 1);
#pragma omp for schedule(static)
        for (int r = 0; r < rows; r++)
            row_dot(W, s, T, x, xs, y, ys, r, tile, td);
    }
#else
    for (int r = 0; r < rows; r++)
        row_dot(W, s, T, x, xs, y, ys, r, s->tile, s->td);
#endif
    count(s, W, T, rows);
}

void dequant_row(const Q35Mat *W, int r, float *y, int8_t *tile)
{
    const uint8_t *p = W->data + (size_t)r * W->row_bytes;
    const int n = W->cols;
    if (W->type == GGML_PTQ1_0) {
        for (int b = 0; b < n / 128; b++) {
            q35_decode_ptq1(p + 28 * b, tile);
            const float d = rd_f16(p + 28 * b + 26);
            for (int i = 0; i < 128; i++)
                y[128 * b + i] = (float)tile[i] * d;
        }
    } else if (W->type == GGML_Q8_0) {
        for (int b = 0; b < n / 32; b++) {
            const float d = rd_f16(p + 34 * b);
            for (int i = 0; i < 32; i++)
                y[32 * b + i] = (float)((const int8_t *)(p + 34 * b + 2))[i] * d;
        }
    } else if (W->type == GGML_F32) {
        memcpy(y, p, (size_t)n * 4);
    } else {
        for (int i = 0; i < n; i++)
            y[i] = rd_f16(p + 2 * i);
    }
}
