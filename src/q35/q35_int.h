/* Internal helpers shared by the q35 sources. */
#ifndef Q35_INT_H
#define Q35_INT_H

#include "q35.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef Q35_FAST_MATH
#define EXPF(x) ((float)exp((double)(x)))
#define LOGF(x) ((float)log((double)(x)))
#define COSF(x) ((float)cos((double)(x)))
#define SINF(x) ((float)sin((double)(x)))
#define POWF(a, b) ((float)pow((double)(a), (double)(b)))
#else
#define EXPF(x) expf(x)
#define LOGF(x) logf(x)
#define COSF(x) cosf(x)
#define SINF(x) sinf(x)
#define POWF(a, b) powf(a, b)
#endif

static inline float silu(float x) { return x / (1.0f + EXPF(-x)); }
static inline float sigm(float x) { return 1.0f / (1.0f + EXPF(-x)); }



float q35_f16_to_f32(uint16_t h);
float rd_f16(const uint8_t *p);
int conv_dim(const Q35Config *c);
int rot_sig(const Q35Rot *R);
size_t log_stride(const Q35Config *c);
uint16_t q35_f32_to_f16(float f);
void attn(const Q35Model *m, Q35State *s, int l, int T);
void dequant_row(const Q35Mat *W, int r, float *y, int8_t *tile);
void ffn(const Q35Model *m, Q35State *s, int l, int T);
void gdn(const Q35Model *m, Q35State *s, int l, int T);
void gdn_ro(const Q35Model *m, Q35State *s, int l, int T);
void l2norm(float *x, int n, float eps);
void linear(const Q35Mat *W, Q35State *s, int T, const float *x, int xs, int *qready, float *y, int ys);
void q35_quant_q8(const float *x, int n, int8_t *q, float *d);
void rmsnorm(float *y, const float *x, const float *w, int n, float eps);
void ro_free(Q35State *s, int L);
void rot_inverse(const Q35Rot *R, float *x, int n);
void state_round(float *x, int n, int fmt);
uint32_t q8_phase(int l, int h, int row, int pos);
void q8_smooth(const float *M, int rows, int cols, float *cs);
float q8_row(const float *x, int n, const float *cs, uint32_t ph, int dither, int8_t *q);
void q8_row_inplace(float *x, int n, const float *cs, uint32_t ph, int dither);
void kv_quant(const float *x, int n, int bits, int8_t *q, uint16_t *sc);
void kv_dequant(const int8_t *q, const uint16_t *sc, int n, int bits, float *y);

#endif
