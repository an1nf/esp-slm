/* BitNet b1.58 engine for .tbin models: ternary GEMV, KV cache, tiered weight store. */
#ifndef TERN_H
#define TERN_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define TERN_MAGIC "TERN"
#define TERN_VERSION 1u

enum {
    TERN_FMT_T2 = 1,
    TERN_FMT_T3 = 2
};

enum {
    TERN_HEAD_F32 = 0,
    TERN_HEAD_I8 = 1
};

enum {
    TERN_KV_F32 = 0,
    TERN_KV_I8 = 1
};

enum {
    TERN_TIER_PSRAM = 0,
    TERN_TIER_FLASH = 1,
    TERN_TIER_SD = 2,
    TERN_NTIERS = 3
};

#define TERN_LINEARS 7

#define TERN_CHUNK_DEFAULT 16384u

typedef struct {
    int fmt, head_fmt;
    int dim, n_layers, n_heads, n_kv, head_dim, inter, vocab, max_seq;
    float rope_theta, eps;
} TernConfig;

typedef struct {
    int tier;
    int r0, r1;
    const uint8_t *data;
    long off;
} TernSeg;

typedef struct {
    int unit;
    int rows, cols, row_bytes;
    float scale;
    int nseg;
    TernSeg seg[TERN_NTIERS];
} TernMat;

typedef struct {
    const float *attn_norm, *attn_sub_norm, *ffn_norm, *ffn_sub_norm;
    TernMat q, k, v, o, gate, up, down;
} TernLayer;

typedef struct {
    TernConfig cfg;
    TernMat head;
    const float *final_norm;
    TernLayer *layers;
    uint8_t *aux;
    size_t aux_bytes;
    uint8_t *arena[TERN_NTIERS];
    size_t placed[TERN_NTIERS];
    size_t max_stream_row;
    FILE *fp;
} TernModel;

typedef struct {
    int kv_fmt;
    float *x, *xn, *q, *k, *v, *att, *o, *g, *u, *h, *scores, *logits;
    int8_t *xq, *tile;
    int32_t *acc;
    void *kcache, *vcache;
    float *kscale, *vscale;
    uint8_t *sbuf;
    size_t sbuf_bytes;
    size_t chunk_bytes;
    FILE *trace;

    void *io;
    const uint8_t *(*io_fetch)(void *io, const TernMat *M, int seg, int r, int n);
    void (*io_release)(void *io, int tier);

    uint64_t bytes_ternary;
    uint64_t bytes_head;
    uint64_t bytes_emb;
    uint64_t bytes_kv;
    uint64_t bytes_aux;
    uint64_t macs;
    uint64_t tier_bytes[TERN_NTIERS];
    uint64_t tier_reads[TERN_NTIERS];
    uint64_t io_errors;
} TernState;

int  tern_open(TernModel *m, const char *path, const char *place_path,
               char *err, size_t errlen);
int  tern_load(TernModel *m, const char *path, char *err, size_t errlen);
void tern_free(TernModel *m);
const char *tern_tier_name(int tier);

int  tern_state_init(TernState *s, const TernModel *m, int kv_fmt,
                     size_t stream_buf);
void tern_state_free(TernState *s);
void tern_state_reset_counters(TernState *s);

const float *tern_forward(const TernModel *m, TernState *s, int token, int pos);

#define TERN_MAX_LANES 128
void tern_forward_batch(const TernModel *m, TernState *const *L, int n,
                        const int *tokens, int pos);

typedef void (*TernPlanFn)(void *ctx, const TernMat *M, int seg, int r, int n);
void tern_plan(const TernModel *m, size_t chunk_bytes, TernPlanFn cb, void *ctx);

int  tern_set_chunk(TernState *s, size_t chunk_bytes);

void    tern_lut_init(void);
int     tern_row_bytes(int fmt, int n);
int     tern_tile_bytes(int n);
void    tern_pack_row(int fmt, const int8_t *w, int n, uint8_t *out);
void    tern_decode_row(int fmt, const uint8_t *packed, int n, int8_t *tile);
int32_t tern_dot_i8(const int8_t *a, const int8_t *b, int n);
void    tern_gemv_rows(int fmt, const uint8_t *data, int rows, int cols,
                       int row_bytes, const int8_t *xq, int8_t *tile,
                       int32_t *acc);
void    tern_gemv_ref(int fmt, const uint8_t *data, int rows, int cols,
                      int row_bytes, const int8_t *xq, int32_t *acc);
float   tern_quant_act(const float *x, int n, int8_t *xq);

#endif
