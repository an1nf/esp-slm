/* Multi-lane ternary GEMV: decode+MAC vs per-lane LUT on base-3 packed bytes (5 trits/byte). */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint8_t POW3[5] = {1, 3, 9, 27, 81};

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* byte b encodes v in [0, 243): b = ceil(256 v / 243); digit j = ((uint8_t)(b * 3^j) * 3) >> 8 */
static uint8_t pack5(const int8_t *t)
{
    int v = 0;
    for (int j = 0; j < 5; j++)
        v = v * 3 + (t[j] + 1);
    return (uint8_t)((v * 256 + 242) / 243);
}

static int digit(uint8_t b, int j)
{
    return (((uint8_t)(b * POW3[j])) * 3) >> 8;
}

static void decode_row(const uint8_t *w, int nb, int8_t *t)
{
    for (int g = 0; g < nb; g++)
        for (int j = 0; j < 5; j++)
            t[g * 5 + j] = (int8_t)(digit(w[g], j) - 1);
}

static void gemv_mac(const uint8_t *W, int R, int K, const int8_t *x, int N, int8_t *tr, int32_t *y)
{
    const int nb = K / 5;
    for (int r = 0; r < R; r++) {
        decode_row(W + (size_t)r * nb, nb, tr);
        for (int n = 0; n < N; n++) {
            const int8_t *xn = x + (size_t)n * K;
            int32_t acc = 0;
            for (int i = 0; i < K; i++)
                acc += (int32_t)tr[i] * xn[i];
            y[(size_t)r * N + n] = acc;
        }
    }
}

/* T[g][b][n] = sum_j trit_j(b) * x[n][5g + j] */
static void lut_build(const int8_t *x, int K, int N, int16_t *T)
{
    const int nb = K / 5;
    int16_t s01[9 * 64], s234[27 * 64];
    for (int g = 0; g < nb; g++) {
        int16_t *Tg = T + (size_t)g * 256 * N;
        for (int a = 0; a < 9; a++)
            for (int n = 0; n < N; n++) {
                const int8_t *xn = x + (size_t)n * K + 5 * g;
                s01[a * N + n] = (int16_t)((a / 3 - 1) * xn[0] + (a % 3 - 1) * xn[1]);
            }
        for (int c = 0; c < 27; c++)
            for (int n = 0; n < N; n++) {
                const int8_t *xn = x + (size_t)n * K + 5 * g;
                s234[c * N + n] = (int16_t)((c / 9 - 1) * xn[2] + (c / 3 % 3 - 1) * xn[3]
                                            + (c % 3 - 1) * xn[4]);
            }
        for (int v = 0; v < 243; v++) {
            const int b = (v * 256 + 242) / 243;
            int16_t *dst = Tg + (size_t)b * N;
            const int16_t *p = s01 + (v / 27) * N, *q = s234 + (v % 27) * N;
            for (int n = 0; n < N; n++)
                dst[n] = (int16_t)(p[n] + q[n]);
        }
    }
}

static void gemv_lut(const uint8_t *W, int R, int K, const int16_t *T, int N, int32_t *y)
{
    const int nb = K / 5;
    int32_t acc[64];
    for (int r = 0; r < R; r++) {
        const uint8_t *w = W + (size_t)r * nb;
        memset(acc, 0, sizeof(acc));
        for (int g = 0; g < nb; g++) {
            const int16_t *e = T + ((size_t)g * 256 + w[g]) * N;
            for (int n = 0; n < N; n++)
                acc[n] += e[n];
        }
        memcpy(y + (size_t)r * N, acc, (size_t)N * sizeof(int32_t));
    }
}

static void fit(const double *xs, const double *ys, int n, double *a, double *b)
{
    double mx = 0, my = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        mx += xs[i] / n;
        my += ys[i] / n;
    }
    for (int i = 0; i < n; i++) {
        sxx += (xs[i] - mx) * (xs[i] - mx);
        sxy += (xs[i] - mx) * (ys[i] - my);
    }
    *b = sxx > 0 ? sxy / sxx : 0;
    *a = my - *b * mx;
}

int main(int argc, char **argv)
{
    int R = 2048, K = 2560, reps = 5, check = 0;
    for (int i = 1; i < argc; i++) {
        const int hv = i + 1 < argc;
        if (!strcmp(argv[i], "--check")) check = 1;
        else if (hv && !strcmp(argv[i], "--rows")) R = atoi(argv[++i]);
        else if (hv && !strcmp(argv[i], "--cols")) K = atoi(argv[++i]);
        else if (hv && !strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: gemvbench [--rows R] [--cols K] [--reps n] [--check]\n");
            return 2;
        }
    }
    K -= K % 5;
    if (R < 1 || K < 5 || reps < 1)
        return 2;
    for (int v = 0; v < 243; v++) {
        int8_t t[5];
        for (int j = 0, x = v; j < 5; j++, x /= 3)
            t[4 - j] = (int8_t)(x % 3 - 1);
        const uint8_t b = pack5(t);
        for (int j = 0; j < 5; j++)
            if (digit(b, j) - 1 != t[j]) {
                printf("base-3 decode identity: FAIL at %d\n", v);
                return 1;
            }
    }
    static const int NS[] = {1, 2, 4, 8, 16, 32, 64};
    const int nn = check ? 4 : 7, NMAX = 64, nb = K / 5;
    uint8_t *W = (uint8_t *)malloc((size_t)R * nb);
    int8_t *x = (int8_t *)malloc((size_t)NMAX * K), *tr = (int8_t *)malloc((size_t)K);
    int16_t *T = (int16_t *)malloc((size_t)nb * 256 * NMAX * sizeof(int16_t));
    int32_t *y1 = (int32_t *)malloc((size_t)R * NMAX * 4), *y2 = (int32_t *)malloc((size_t)R * NMAX * 4);
    if (!W || !x || !tr || !T || !y1 || !y2)
        return 1;
    uint32_t s = 1;
    for (size_t i = 0; i < (size_t)R * nb; i++) {
        int8_t t[5];
        for (int j = 0; j < 5; j++) {
            s = s * 1664525u + 1013904223u;
            const uint32_t u = s >> 24;
            t[j] = (int8_t)(u < 82 ? -1 : u < 174 ? 0 : 1);
        }
        W[i] = pack5(t);
    }
    for (size_t i = 0; i < (size_t)NMAX * K; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (int8_t)((int)(s >> 24) - 128);
        if (x[i] == -128)
            x[i] = -127;
    }
    printf("gemv %d x %d, 5 trits/byte (%.2f MB)\n", R, K, (double)R * nb / 1e6);
    printf("  N   mac ms   lut ms  (build)   mac GMAC/s  lut GMAC/s\n");
    double xs[7], tm[7], tl[7];
    int ok = 1;
    for (int k = 0; k < nn; k++) {
        const int N = NS[k];
        double bm = 1e30, bl = 1e30, bb = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now_s();
            gemv_mac(W, R, K, x, N, tr, y1);
            const double t1 = now_s();
            lut_build(x, K, N, T);
            const double t2 = now_s();
            gemv_lut(W, R, K, T, N, y2);
            const double t3 = now_s();
            if (t1 - t0 < bm) bm = t1 - t0;
            if (t3 - t1 < bl) bl = t3 - t1;
            if (t2 - t1 < bb) bb = t2 - t1;
        }
        ok &= !memcmp(y1, y2, (size_t)R * N * 4);
        const double macs = (double)R * K * N;
        printf("%3d %8.3f %8.3f (%6.3f) %10.2f %11.2f\n", N, bm * 1e3, bl * 1e3, bb * 1e3,
               macs / bm / 1e9, macs / bl / 1e9);
        xs[k] = N;
        tm[k] = bm * 1e3;
        tl[k] = bl * 1e3;
    }
    double am, km, al, kl;
    fit(xs, tm, nn, &am, &km);
    fit(xs, tl, nn, &al, &kl);
    printf("fit T(N) = alpha + kappa N: mac %.3f + %.3f N ms, lut %.3f + %.3f N ms\n", am, km, al, kl);
    if (km > kl && al > am)
        printf("crossover: lut faster for N > %.1f\n", (al - am) / (km - kl));
    else if (km < kl && am > al)
        printf("crossover: mac faster for N > %.1f\n", (am - al) / (kl - km));
    else
        printf("crossover: none (%s faster for all N)\n", km <= kl && am <= al ? "mac" : "lut");
    printf("mac = lut bit-exact: %s\n", ok ? "OK" : "FAIL");
    free(W); free(x); free(tr); free(T); free(y1); free(y2);
    return ok ? 0 : 1;
}
