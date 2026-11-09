#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "hx.h"

static int8_t LUT5[256][5];
static int lut_ready;

double hx_now_s(void)
{
    return (double)esp_timer_get_time() * 1e-6;
}

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

/* PTQ1_0 block: qs[16] x 5 trits, qs[8] x 5, qh[2] x 4, f16 scale. Same order as q35. */
void hx_decode_ptq1(const uint8_t *b, int8_t *t)
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

void hx_decode_row(const uint8_t *row, int cols, int8_t *out)
{
    for (int b = 0; b < cols / PTQ1_TRITS; b++)
        hx_decode_ptq1(row + (size_t)b * PTQ1_BLOCK, out + (size_t)b * PTQ1_TRITS);
}

static uint8_t enc5(const int *d)
{
    int v = 0;
    for (int j = 0; j < 5; j++)
        v = v * 3 + d[j];
    return (uint8_t)((v * 256 + 242) / 243);
}

void hx_pack_ptq1(const int8_t *t, uint16_t d, uint8_t *b)
{
    int dg[5];
    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 5; n++)
            dg[n] = t[16 * n + m] + 1;
        b[m] = enc5(dg);
    }
    for (int m = 0; m < 8; m++) {
        for (int n = 0; n < 5; n++)
            dg[n] = t[80 + 8 * n + m] + 1;
        b[16 + m] = enc5(dg);
    }
    for (int h = 0; h < 2; h++) {
        for (int n = 0; n < 4; n++)
            dg[n] = t[120 + 2 * n + h] + 1;
        dg[4] = 1;
        b[24 + h] = enc5(dg);
    }
    b[26] = (uint8_t)(d & 0xff);
    b[27] = (uint8_t)(d >> 8);
}

uint8_t hx_pack5(const int8_t *t)
{
    int dg[5];
    for (int j = 0; j < 5; j++)
        dg[j] = t[j] + 1;
    return enc5(dg);
}

/* Lane-major activations: xl[16 g k + n] for lane group g. */
void hx_gemv_ref(const int8_t *w, const int8_t *xl, int k, int lanes, int32_t *y)
{
    for (int g = 0; g < (lanes + 15) / 16; g++)
        for (int n = 0; n < 16 && 16 * g + n < lanes; n++) {
            int32_t acc = 0;
            for (int i = 0; i < k; i++)
                acc += (int32_t)w[i] * xl[(size_t)g * 16 * k + (size_t)16 * i + n];
            y[16 * g + n] = acc;
        }
}

/* T[g][b][n] = sum_j trit_j(b) x_n[5g + j], lanes <= 16 from group 0 */
void hx_lut_build(const int8_t *xl, int k, int lanes, int16_t *T)
{
    if (!lut_ready)
        lut_init();
    for (int g = 0; g < k / 5; g++)
        for (int b = 0; b < 256; b++) {
            int16_t *e = T + ((size_t)g * 256 + b) * lanes;
            for (int n = 0; n < lanes; n++) {
                int s = 0;
                for (int j = 0; j < 5; j++)
                    s += LUT5[b][j] * xl[(size_t)16 * (5 * g + j) + n];
                e[n] = (int16_t)s;
            }
        }
}

void hx_gemv_lut(const uint8_t *w5, const int16_t *T, int k, int lanes, int32_t *y)
{
    int32_t acc[16] = {0};
    for (int g = 0; g < k / 5; g++) {
        const int16_t *e = T + ((size_t)g * 256 + w5[g]) * lanes;
        for (int n = 0; n < lanes; n++)
            acc[n] += e[n];
    }
    memcpy(y, acc, (size_t)lanes * sizeof(int32_t));
}

static uint32_t rng = 1;
static int8_t rtrit(void)
{
    rng = rng * 1664525u + 1013904223u;
    const uint32_t u = rng >> 24;
    return (int8_t)(u < 82 ? -1 : u < 174 ? 0 : 1);
}

static int8_t rbyte(void)
{
    rng = rng * 1664525u + 1013904223u;
    const int v = (int)(rng >> 24) - 128;
    return (int8_t)(v == -128 ? -127 : v);
}

void bench_kernels(hx_meas_t *m)
{
    enum { K = 2560, R = 48, REPS = 4 };
    const int nb = K / PTQ1_TRITS;
    uint8_t *pk = heap_caps_malloc((size_t)R * nb * PTQ1_BLOCK, MALLOC_CAP_SPIRAM);
    int8_t *w = heap_caps_aligned_alloc(16, (size_t)R * K + 16, MALLOC_CAP_INTERNAL);
    int8_t *xl = heap_caps_aligned_alloc(16, (size_t)2 * 16 * K + 32, MALLOC_CAP_INTERNAL);
    int8_t *x0 = heap_caps_aligned_alloc(16, (size_t)K + 16, MALLOC_CAP_INTERNAL);
    int32_t *y = heap_caps_aligned_alloc(16, (size_t)R * 32 * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    int32_t *yr = heap_caps_malloc((size_t)32 * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    if (!pk || !w || !xl || !x0 || !y || !yr) {
        printf("kernels: out of memory\n");
        return;
    }
    int8_t t[PTQ1_TRITS], t2[PTQ1_TRITS];
    uint8_t blk[PTQ1_BLOCK];
    int rt = 0;
    for (int it = 0; it < 64; it++) {
        for (int i = 0; i < PTQ1_TRITS; i++)
            t[i] = rtrit();
        hx_pack_ptq1(t, 0x3c00, blk);
        hx_decode_ptq1(blk, t2);
        rt += memcmp(t, t2, PTQ1_TRITS) != 0;
    }
    printf("PTQ1_0 pack/decode round trip: %s\n", rt ? "FAIL" : "OK");
    for (int r = 0; r < R; r++)
        for (int b = 0; b < nb; b++) {
            for (int i = 0; i < PTQ1_TRITS; i++)
                t[i] = rtrit();
            hx_pack_ptq1(t, 0x3c00, pk + ((size_t)r * nb + b) * PTQ1_BLOCK);
        }
    for (size_t i = 0; i < (size_t)2 * 16 * K + 32; i++)
        xl[i] = rbyte();
    for (int i = 0; i < K; i++)
        x0[i] = xl[16 * i];

    double t0 = hx_now_s();
    for (int rep = 0; rep < REPS; rep++)
        for (int r = 0; r < R; r++)
            hx_decode_row(pk + (size_t)r * nb * PTQ1_BLOCK, K, w + (size_t)r * K);
    double dt = hx_now_s() - t0;
    m->decode_mbps = (double)REPS * R * nb * PTQ1_BLOCK / dt / 1e6;
    printf("decode PTQ1_0 (PSRAM -> SRAM, C LUT): %.1f MB/s packed, %.1f Mtrit/s\n", m->decode_mbps,
           (double)REPS * R * K / dt / 1e6);

    int bad = 0;
    t0 = hx_now_s();
    for (int rep = 0; rep < REPS; rep++)
        for (int r = 0; r < R; r++)
            y[r] = hx_dot_s8(w + (size_t)r * K, x0, K);
    dt = hx_now_s() - t0;
    m->gmacs_n1 = (double)REPS * R * K / dt / 1e9;
    for (int r = 0; r < R; r++) {
        int32_t ref = 0;
        for (int i = 0; i < K; i++)
            ref += (int32_t)w[(size_t)r * K + i] * x0[i];
        bad += ref != y[r];
    }
    printf("gemv N=1  (xacc dot):        %.2f GMAC/s, check %s\n", m->gmacs_n1, bad ? "FAIL" : "OK");

    for (int lanes = 16; lanes <= 32; lanes += 16) {
        t0 = hx_now_s();
        for (int rep = 0; rep < REPS; rep++)
            for (int r = 0; r < R; r++)
                for (int g = 0; g < lanes / 16; g++)
                    hx_gemv16_s8(w + (size_t)r * K, xl + (size_t)g * 16 * K, K,
                                 y + (size_t)r * 32 + 16 * g);
        dt = hx_now_s() - t0;
        const double gm = (double)REPS * R * K * lanes / dt / 1e9;
        bad = 0;
        for (int r = 0; r < R; r++) {
            hx_gemv_ref(w + (size_t)r * K, xl, K, lanes, yr);
            bad += memcmp(yr, y + (size_t)r * 32, (size_t)lanes * 4) != 0;
        }
        if (lanes == 16)
            m->gmacs_n16 = gm;
        else
            m->gmacs_n32 = gm;
        printf("gemv N=%d (qacc lane-major): %.2f GMAC/s, check %s%s\n", lanes, gm, bad ? "FAIL" : "OK",
               lanes == 16 ? (gm >= 4.0 ? "  [>= 4 GMAC/s: PASS]" : "  [>= 4 GMAC/s: FAIL]") : "");
    }

    t0 = hx_now_s();
    for (int r = 0; r < R; r++) {
        hx_decode_row(pk + (size_t)r * nb * PTQ1_BLOCK, K, w);
        hx_gemv16_s8(w, xl, K, y);
    }
    dt = hx_now_s() - t0;
    printf("decode + gemv N=16 per row:  %.2f GMAC/s effective\n", (double)R * K * 16 / dt / 1e9);

    enum { KL = 640, RL = 64 };
    uint8_t *w5 = heap_caps_malloc((size_t)RL * KL / 5, MALLOC_CAP_INTERNAL);
    int16_t *T = heap_caps_malloc((size_t)KL / 5 * 256 * 8 * sizeof(int16_t), MALLOC_CAP_DEFAULT);
    int32_t *yl = heap_caps_aligned_alloc(16, (size_t)2 * RL * 16 * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    if (!w5 || !T || !yl) {
        printf("lut: out of memory\n");
        return;
    }
    for (int i = 0; i < RL * KL / 5; i++) {
        int8_t q[5];
        for (int j = 0; j < 5; j++)
            q[j] = rtrit();
        w5[i] = hx_pack5(q);
    }
    printf("lut vs mac, %d x %d rows (ms per call)\n  N   lut     lut build   mac(pie)\n", RL, KL);
    double xs[4], tl[4], tm[4];
    int8_t *wd = w;
    bad = 0;
    for (int r = 0; r < RL; r++)
        for (int g = 0; g < KL / 5; g++)
            for (int j = 0; j < 5; j++)
                wd[(size_t)r * KL + 5 * g + j] = LUT5[w5[(size_t)r * KL / 5 + g]][j];
    for (int k = 0; k < 4; k++) {
        const int lanes = 1 << k;
        const double a = hx_now_s();
        hx_lut_build(xl, KL, lanes, T);
        const double b = hx_now_s();
        for (int r = 0; r < RL; r++)
            hx_gemv_lut(w5 + (size_t)r * KL / 5, T, KL, lanes, yl + (size_t)r * 16);
        const double c = hx_now_s();
        for (int r = 0; r < RL; r++)
            hx_gemv16_s8(wd + (size_t)r * KL, xl, KL, yl + (size_t)RL * 16 + (size_t)r * 16);
        const double d = hx_now_s();
        for (int r = 0; r < RL; r++)
            bad += memcmp(yl + (size_t)r * 16, yl + (size_t)RL * 16 + (size_t)r * 16, (size_t)lanes * 4) != 0;
        xs[k] = lanes;
        tl[k] = (c - a) * 1e3;
        tm[k] = (d - c) * 1e3;
        printf("%3d %7.3f %9.3f %9.3f\n", lanes, tl[k], (b - a) * 1e3, tm[k]);
    }
    double mx = 0, ml = 0, mm = 0, sxx = 0, sl = 0, sm = 0;
    for (int k = 0; k < 4; k++) {
        mx += xs[k] / 4;
        ml += tl[k] / 4;
        mm += tm[k] / 4;
    }
    for (int k = 0; k < 4; k++) {
        sxx += (xs[k] - mx) * (xs[k] - mx);
        sl += (xs[k] - mx) * (tl[k] - ml);
        sm += (xs[k] - mx) * (tm[k] - mm);
    }
    const double kl = sl / sxx, km = sm / sxx, al = ml - kl * mx, am = mm - km * mx;
    printf("fit: lut %.3f + %.3f N ms, mac %.3f + %.3f N ms (mac covers 16 lanes per call)\n", al, kl, am, km);
    printf("lut = mac: %s\n", bad ? "FAIL" : "OK");
    heap_caps_free(w5);
    heap_caps_free(T);
    heap_caps_free(yl);
    heap_caps_free(pk);
    heap_caps_free(w);
    heap_caps_free(xl);
    heap_caps_free(x0);
    heap_caps_free(y);
    heap_caps_free(yr);
}
