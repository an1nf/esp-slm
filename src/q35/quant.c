/* State quantizers: Weyl dither, key smoothing, decay-based 16-bit heads. */
#include "q35_int.h"

/* Weyl phase: per-element offset plus pos * (phi - 1) in 32-bit fixed point. */
#define WEYL_A 0x9E3779B9u
#define WEYL_R 0x27D4EB2Fu
#define WEYL_E 0x85EBCA6Bu

uint32_t q8_phase(int l, int h, int row, int pos)
{
    const uint32_t id = ((uint32_t)l * 1031u + (uint32_t)h) * 4099u + (uint32_t)row;
    return id * WEYL_R + (uint32_t)pos * WEYL_A;
}

/* Per-key-column smoothing: cs_i ~ sqrt(max_j |M_ji|), geometric mean 1. */
void q8_smooth(const float *M, int rows, int cols, float *cs)
{
    double lg = 0.0;
    int nz = 0;
    for (int i = 0; i < cols; i++) {
        float c = 0.0f;
        for (int j = 0; j < rows; j++) {
            const float a = fabsf(M[(size_t)j * cols + i]);
            if (a > c)
                c = a;
        }
        cs[i] = c > 0.0f ? sqrtf(c) : 0.0f;
        if (cs[i] > 0.0f) {
            lg += log((double)cs[i]);
            nz++;
        }
    }
    const float gm = nz ? (float)exp(lg / nz) : 1.0f;
    for (int i = 0; i < cols; i++)
        cs[i] = cs[i] > 0.0f ? cs[i] / gm : 1.0f;
}

/* Symmetric int8 row quantizer; cs divides columns, dither adds a Weyl offset. */
float q8_row(const float *x, int n, const float *cs, uint32_t ph, int dither, int8_t *q)
{
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = fabsf(cs ? x[i] / cs[i] : x[i]);
        if (a > amax)
            amax = a;
    }
    if (amax == 0.0f) {
        memset(q, 0, (size_t)n);
        return 0.0f;
    }
    const float id = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        const float y = (cs ? x[i] / cs[i] : x[i]) * id;
        int v;
        if (dither) {
            const uint32_t u = (ph + (uint32_t)i * WEYL_E) >> 8;
            v = (int)floorf(y + (float)u * (1.0f / 16777216.0f));
        } else {
            v = (int)rintf(y);
        }
        q[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
    return amax / 127.0f;
}

void q8_row_inplace(float *x, int n, const float *cs, uint32_t ph, int dither)
{
    int8_t q[512];
    if (n > 512)
        return;
    const float d = q8_row(x, n, cs, ph, dither, q);
    for (int i = 0; i < n; i++)
        x[i] = (float)q[i] * d * (cs ? cs[i] : 1.0f);
}

/* Decay gain per v-head: G = 1 / expm1(2 s), s = -a * softplus(dt). */
static double head_gain(const Q35Layer *Ly, int h)
{
    const double dt = Ly->ssm_dt[h];
    const double sp = dt > 20.0 ? dt : log1p(exp(dt));
    double sdec = -(double)Ly->ssm_a[h] * sp;
    if (sdec < 1e-6)
        sdec = 1e-6;
    return 1.0 / expm1(2.0 * sdec);
}

/* Keep the top hi_frac of v-heads by decay gain in 16 bits (water-filling with two levels). */
int q35_pick_hi(const Q35Model *m, Q35State *s)
{
    const Q35Config *c = &m->c;
    const int Hv = c->v_heads, L = c->n_layer;
    if (s->hi)
        return 0;
    s->hi = (uint8_t *)calloc((size_t)L * Hv, 1);
    if (!s->hi)
        return -1;
    int nrec = 0;
    for (int l = 0; l < L; l++)
        nrec += m->L[l].recurrent;
    const int tot = nrec * Hv;
    int want = (int)floor(s->hi_frac * tot + 0.5);
    if (want <= 0)
        return 0;
    double *g = (double *)malloc((size_t)tot * sizeof(double));
    int *id = (int *)malloc((size_t)tot * sizeof(int));
    if (!g || !id) {
        free(g);
        free(id);
        return -1;
    }
    int n = 0;
    for (int l = 0; l < L; l++)
        if (m->L[l].recurrent)
            for (int h = 0; h < Hv; h++) {
                g[n] = head_gain(&m->L[l], h);
                id[n++] = l * Hv + h;
            }
    for (int k = 0; k < want && k < n; k++) {
        int b = k;
        for (int i = k + 1; i < n; i++)
            if (g[i] > g[b] || (g[i] == g[b] && id[i] < id[b]))
                b = i;
        const double tg = g[k];
        const int ti = id[k];
        g[k] = g[b];
        id[k] = id[b];
        g[b] = tg;
        id[b] = ti;
        s->hi[id[k]] = 1;
    }
    free(g);
    free(id);
    return 0;
}

/* Checks: dither is unbiased, breaks the IIR deadband, smoothing helps outlier columns. */
int q35_quant_selftest(FILE *f)
{
    enum { N = 128, P = 4096 };
    float x[N], y[N], bias_d[N] = {0}, bias_r[N] = {0};
    int fails = 0;
    uint32_t r = 12345u;
    for (int i = 0; i < N; i++) {
        r = r * 1664525u + 1013904223u;
        x[i] = ((float)(r >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }
    x[0] = 1.0f;
    const float step = 1.0f / 127.0f;
    for (int p = 0; p < P; p++) {
        memcpy(y, x, sizeof(x));
        q8_row_inplace(y, N, NULL, q8_phase(0, 0, 0, p), 1);
        for (int i = 0; i < N; i++)
            bias_d[i] += (y[i] - x[i]) / P;
    }
    memcpy(y, x, sizeof(x));
    q8_row_inplace(y, N, NULL, 0, 0);
    float md = 0.0f, mr = 0.0f;
    for (int i = 0; i < N; i++) {
        bias_r[i] = y[i] - x[i];
        if (fabsf(bias_d[i]) > md) md = fabsf(bias_d[i]);
        if (fabsf(bias_r[i]) > mr) mr = fabsf(bias_r[i]);
    }
    int ok = md < 0.02f * step && mr > 0.3f * step;
    fprintf(f, "  dither bias: max %.3f step (round-to-nearest %.3f): %s\n", md / step, mr / step,
            ok ? "OK" : "FAIL");
    fails += !ok;

    const float g = 0.995f, A = 1.0f;
    double err[2] = {0.0, 0.0};
    for (int d = 0; d < 2; d++) {
        float s2[2] = {A, 0.5f * A};
        double tv = 0.5 * A;
        for (int t = 0; t < 3000; t++) {
            s2[0] = g * s2[0] + (1.0f - g) * A;
            s2[1] = g * s2[1];
            tv *= g;
            float row[32] = {0};
            row[0] = s2[0];
            row[1] = s2[1];
            q8_row_inplace(row, 32, NULL, q8_phase(0, 0, 1, t), d);
            s2[0] = row[0];
            s2[1] = row[1];
            if (t >= 2500)
                err[d] += fabs(s2[1] - tv) / 500.0;
        }
    }
    ok = err[0] > 0.3 * A && err[1] < 0.1 * A;
    fprintf(f, "  IIR deadband (gamma %.3f): mean |err| %.3f round, %.3f dither: %s\n", g, err[0],
            err[1], ok ? "OK" : "FAIL");
    fails += !ok;

    enum { R = 64, C = 128 };
    static float M[R * C], Q[R * C];
    float cs[C], q[C];
    for (int j = 0; j < R; j++)
        for (int i = 0; i < C; i++) {
            r = r * 1664525u + 1013904223u;
            M[j * C + i] = (((float)(r >> 8) / 16777216.0f) - 0.5f) * (i % 16 == 3 ? 40.0f : 1.0f);
        }
    for (int i = 0; i < C; i++) {
        r = r * 1664525u + 1013904223u;
        q[i] = ((float)(r >> 8) / 16777216.0f) - 0.5f;
    }
    double e[2] = {0.0, 0.0};
    for (int sm = 0; sm < 2; sm++) {
        memcpy(Q, M, sizeof(M));
        if (sm)
            q8_smooth(Q, R, C, cs);
        for (int j = 0; j < R; j++)
            q8_row_inplace(Q + j * C, C, sm ? cs : NULL, 0, 0);
        for (int j = 0; j < R; j++) {
            double a = 0.0, b = 0.0;
            for (int i = 0; i < C; i++) {
                a += (double)M[j * C + i] * q[i];
                b += (double)Q[j * C + i] * q[i];
            }
            e[sm] += (a - b) * (a - b) / R;
        }
    }
    ok = e[1] < 0.8 * e[0];
    fprintf(f, "  key smoothing: dot mse %.3g plain, %.3g smoothed: %s\n", e[0], e[1],
            ok ? "OK" : "FAIL");
    fails += !ok;

    fprintf(f, "quant selftest: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
