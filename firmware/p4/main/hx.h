#ifndef HX_H
#define HX_H

#include <stddef.h>
#include <stdint.h>

#define PTQ1_BLOCK 28
#define PTQ1_TRITS 128

typedef struct {
    double sd_mbps[4];
    double sd_lat_ms[4];
    int sd_khz[4];
    int sd_n;
    double psram_mbps;
    double decode_mbps;
    double gmacs_n1, gmacs_n16, gmacs_n32;
} hx_meas_t;

void hx_decode_ptq1(const uint8_t *blk, int8_t *t128);
void hx_decode_row(const uint8_t *row, int cols, int8_t *out);
void hx_pack_ptq1(const int8_t *t128, uint16_t d, uint8_t *blk);

int32_t hx_dot_s8(const int8_t *a, const int8_t *b, int k);
void hx_gemv16_s8(const int8_t *w, const int8_t *xl, int k, int32_t *y16);

void hx_gemv_ref(const int8_t *w, const int8_t *xl, int k, int lanes, int32_t *y);
void hx_lut_build(const int8_t *xl, int k, int lanes, int16_t *T);
void hx_gemv_lut(const uint8_t *w5, const int16_t *T, int k, int lanes, int32_t *y);
uint8_t hx_pack5(const int8_t *t);

void bench_mem(hx_meas_t *m);
void bench_kernels(hx_meas_t *m);
void bench_sd(hx_meas_t *m);
void calibrate(const hx_meas_t *m);

double hx_now_s(void);

#endif
