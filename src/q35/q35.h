/* Qwen3.5 hybrid (Gated DeltaNet + gated attention) inference for Ternary Bonsai 2 GGUF. */
#ifndef Q35_H
#define Q35_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "gguf.h"

typedef struct {
    int block;
    const float *signs;
    int perm_hd, perm_nk, perm_rep;
} Q35Rot;

typedef struct {
    int type;
    int rows, cols;
    size_t row_bytes;
    const uint8_t *data;
    float scale;
    int has_scale;
    int kind;
    int layer;
    Q35Rot rot;
} Q35Mat;

enum {
    Q35_K_EMBD, Q35_K_HEAD, Q35_K_ATTN, Q35_K_GDN, Q35_K_GDN_F32, Q35_K_FFN,
    Q35_K_N
};

typedef struct {
    int recurrent;
    const float *attn_norm, *post_norm;
    Q35Mat wq, wk, wv, wo;
    const float *q_norm, *k_norm;
    Q35Mat wqkv, wz, alpha, beta, ssm_out;
    const float *conv, *ssm_a, *ssm_dt, *ssm_norm;
    Q35Mat gate, up, down;
} Q35Layer;

typedef struct {
    int n_embd, n_layer, n_ff, vocab;
    int n_head, n_kv, head_dim, n_rot;
    int d_conv, k_heads, v_heads, d_k, d_v;
    float eps, rope_base;
    int sect[4];
    int n_nextn;
} Q35Config;

typedef struct {
    Q35Config c;
    Q35Mat embd, head;
    Q35Rot embd_inv;
    int hd_version, hd_folded;
    const float *out_norm;
    Q35Layer *L;
    void **blobs;
    int nblobs, capblobs;
    uint64_t bytes_kind[Q35_K_N];
} Q35Model;

enum { Q35_STATE_F32, Q35_STATE_F16, Q35_STATE_BF16, Q35_STATE_I8,
       Q35_STATE_RO32, Q35_STATE_RO8, Q35_STATE_RO16 };

/* Decoding state. Recurrent state and conv tails per GDN layer, KV per attention layer,
   scratch for up to max_T tokens per pass, tree bookkeeping and read-only (ro*) buffers. */
typedef struct {
    int max_seq, max_T, kv_f16;
    int state_fmt;
    float noise_eps;
    int n_past;
    float **conv, **ssm;
    void **kc, **vc;

    float *x, *xn, *o, *big, *z, *att, *kbuf, *vbuf, *g, *u, *bt, *al;
    float *co, *scores, *logits, *rc, *rs;
    int8_t *aq, *tile;
    float *ad, *td, *hx;
    int act_gen, aq_gen, aq_sig, hx_gen, hx_sig;
    int nthreads;
    size_t tile_stride;

    uint64_t bytes_kind[Q35_K_N];
    uint64_t macs;
    FILE *trace;
    FILE *dbg;
    FILE *hdump;
    int n_layers_alloc;

    float **lg;
    float *S_tmp, *C_tmp;
    int n_pend;
    int last_T;
    int *pend;

    int tree, *tpar, *tdep, *slots;
    float *S_stk, *C_stk;

    int ro_rmax, ro_cap;
    int8_t **ro_b;
    float **ro_bs;
    float **ro_lgam;
    float **ro_U, **ro_K, **ro_w;
    int *ro_n;
    float **ro_ut, **ro_lG;
    float *ro_q, *ro_ak, *ro_aq;
    uint64_t ro_mat;
} Q35State;

/* Load a Ternary Bonsai 2 GGUF (PTQ1_0 weights). Returns 0 or -1 with err set. */
int  q35_load(Q35Model *m, const char *path, char *err, size_t errlen);
void q35_free(Q35Model *m);
void q35_print_config(const Q35Model *m, FILE *f);

int  q35_state_init(Q35State *s, const Q35Model *m, int max_seq, int max_T,
                    int kv_f16);
void q35_state_free(Q35State *s);
void q35_reset(Q35State *s, const Q35Model *m);

/* Evaluate T tokens and commit them. */
const float *q35_forward(const Q35Model *m, Q35State *s, const int *tok, int T,
                         int all_logits);

/* Evaluate a chain without committing; q35_commit keeps the first n (GDN replay is deferred). */
const float *q35_eval(const Q35Model *m, Q35State *s, const int *tok, int T,
                      int all_logits);
void q35_commit(Q35State *s, int n);

/* Evaluate N tree nodes (parent[i] < i, -1 = root child) in one pass; commit one root path. */
const float *q35_eval_tree(const Q35Model *m, Q35State *s, const int *tok,
                           const int *parent, int N);

int q35_commit_path(const Q35Model *m, Q35State *s, const int *path, int n);

uint16_t q35_f32_to_f16(float f);
float    q35_f16_to_f32(uint16_t h);
void     q35_decode_ptq1(const uint8_t *blk, int8_t *t128);
void     q35_quant_q8(const float *x, int n, int8_t *q, float *d);

#endif
