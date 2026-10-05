/* GGUF v2/v3 reader: metadata, tensor table and positioned reads. */
#ifndef GGUF_H
#define GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum {
    GGUF_T_UINT8 = 0, GGUF_T_INT8 = 1, GGUF_T_UINT16 = 2, GGUF_T_INT16 = 3,
    GGUF_T_UINT32 = 4, GGUF_T_INT32 = 5, GGUF_T_FLOAT32 = 6, GGUF_T_BOOL = 7,
    GGUF_T_STRING = 8, GGUF_T_ARRAY = 9, GGUF_T_UINT64 = 10, GGUF_T_INT64 = 11,
    GGUF_T_FLOAT64 = 12
};

enum {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8, GGML_I8 = 24, GGML_I16 = 25,
    GGML_I32 = 26, GGML_I64 = 27, GGML_F64 = 28, GGML_BF16 = 30,
    GGML_PQ2_0 = 142, GGML_PTQ1_0 = 143
};

typedef struct {
    char *key;
    uint32_t type;
    union { uint64_t u; int64_t i; double f; } v;
    char *str;
    uint32_t arr_type;
    uint64_t arr_n;
    long arr_pos;
} GGUFKV;

typedef struct {
    char *name;
    uint32_t type;
    int n_dims;
    int64_t ne[4];
    uint64_t off;
    uint64_t nbytes;
} GGUFTensor;

typedef struct {
    FILE *fp;
    uint32_t version, alignment;
    uint64_t n_kv, n_tensors;
    GGUFKV *kv;
    GGUFTensor *t;
    uint64_t data_off;
} GGUF;

int  gguf_open(GGUF *g, const char *path, char *err, size_t errlen);
void gguf_close(GGUF *g);

const GGUFKV *gguf_kv(const GGUF *g, const char *key);

int gguf_u32(const GGUF *g, const char *key, uint32_t *out);
int gguf_f32(const GGUF *g, const char *key, float *out);
const char *gguf_str(const GGUF *g, const char *key);

int gguf_arr_i32(const GGUF *g, const char *key, int32_t *out, int max);

int gguf_arr_str(const GGUF *g, const char *key, char ***out);

const GGUFTensor *gguf_tensor(const GGUF *g, const char *name);

size_t gguf_type_bytes(uint32_t type, int64_t n);
const char *gguf_type_name(uint32_t type);

int gguf_read(const GGUF *g, const GGUFTensor *t, uint64_t from, void *dst,
              size_t nbytes);

#endif
