/* GGUF v2/v3 reader. */
#include "gguf.h"

#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n)
{
    return fread(p, 1, n, f) == n ? 0 : -1;
}

static int rd_u32(FILE *f, uint32_t *v)
{
    uint8_t b[4];
    if (rd(f, b, 4))
        return -1;
    *v = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return 0;
}

static int rd_u64(FILE *f, uint64_t *v)
{
    uint32_t lo, hi;
    if (rd_u32(f, &lo) || rd_u32(f, &hi))
        return -1;
    *v = (uint64_t)hi << 32 | lo;
    return 0;
}

static int rd_str(FILE *f, char **out)
{
    uint64_t n;
    if (rd_u64(f, &n) || n > (1u << 30))
        return -1;
    if (!out)
        return fseek(f, (long)n, SEEK_CUR);
    char *s = (char *)malloc((size_t)n + 1);
    if (!s || rd(f, s, (size_t)n)) {
        free(s);
        return -1;
    }
    s[n] = 0;
    *out = s;
    return 0;
}

static size_t scalar_size(uint32_t t)
{
    switch (t) {
    case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL: return 1;
    case GGUF_T_UINT16: case GGUF_T_INT16: return 2;
    case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
    case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
    default: return 0;
    }
}

static int rd_scalar(FILE *f, uint32_t t, GGUFKV *kv)
{
    uint8_t b[8] = {0};
    const size_t n = scalar_size(t);
    if (!n || rd(f, b, n))
        return -1;
    uint64_t u = 0;
    for (size_t i = 0; i < n; i++)
        u |= (uint64_t)b[i] << (8 * i);
    switch (t) {
    case GGUF_T_INT8: kv->v.i = (int8_t)u; break;
    case GGUF_T_INT16: kv->v.i = (int16_t)u; break;
    case GGUF_T_INT32: kv->v.i = (int32_t)u; break;
    case GGUF_T_INT64: kv->v.i = (int64_t)u; break;
    case GGUF_T_FLOAT32: {
        uint32_t w = (uint32_t)u;
        float x;
        memcpy(&x, &w, 4);
        kv->v.f = x;
        break;
    }
    case GGUF_T_FLOAT64: memcpy(&kv->v.f, &u, 8); break;
    default: kv->v.u = u; break;
    }
    return 0;
}

static int skip_array(FILE *f, uint32_t t, uint64_t n)
{
    if (t == GGUF_T_STRING) {
        for (uint64_t i = 0; i < n; i++)
            if (rd_str(f, NULL))
                return -1;
        return 0;
    }
    const size_t s = scalar_size(t);
    if (!s)
        return -1;
    return fseek(f, (long)(s * n), SEEK_CUR);
}

size_t gguf_type_bytes(uint32_t type, int64_t n)
{
    switch (type) {
    case GGML_F32: case GGML_I32: return (size_t)n * 4;
    case GGML_F16: case GGML_BF16: case GGML_I16: return (size_t)n * 2;
    case GGML_I8: return (size_t)n;
    case GGML_I64: case GGML_F64: return (size_t)n * 8;
    case GGML_Q8_0: return n % 32 ? 0 : (size_t)(n / 32) * 34;
    case GGML_PTQ1_0: return n % 128 ? 0 : (size_t)(n / 128) * 28;
    case GGML_PQ2_0: return n % 128 ? 0 : (size_t)(n / 128) * 34;
    default: return 0;
    }
}

const char *gguf_type_name(uint32_t type)
{
    switch (type) {
    case GGML_F32: return "F32";
    case GGML_F16: return "F16";
    case GGML_BF16: return "BF16";
    case GGML_Q8_0: return "Q8_0";
    case GGML_I8: return "I8";
    case GGML_I16: return "I16";
    case GGML_I32: return "I32";
    case GGML_I64: return "I64";
    case GGML_F64: return "F64";
    case GGML_PTQ1_0: return "PTQ1_0";
    case GGML_PQ2_0: return "PQ2_0";
    default: return "?";
    }
}

int gguf_open(GGUF *g, const char *path, char *err, size_t errlen)
{
    memset(g, 0, sizeof(*g));
    g->alignment = 32;
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, errlen, "cannot open %s", path);
        return -1;
    }
    g->fp = f;
    char magic[4];
    if (rd(f, magic, 4) || memcmp(magic, "GGUF", 4) || rd_u32(f, &g->version)
        || (g->version != 2 && g->version != 3) || rd_u64(f, &g->n_tensors)
        || rd_u64(f, &g->n_kv) || g->n_kv > 100000 || g->n_tensors > 1000000) {
        snprintf(err, errlen, "%s: not a GGUF v2/v3 file", path);
        gguf_close(g);
        return -1;
    }
    g->kv = (GGUFKV *)calloc((size_t)g->n_kv + 1, sizeof(GGUFKV));
    g->t = (GGUFTensor *)calloc((size_t)g->n_tensors + 1, sizeof(GGUFTensor));
    if (!g->kv || !g->t) {
        snprintf(err, errlen, "out of memory");
        gguf_close(g);
        return -1;
    }
    for (uint64_t i = 0; i < g->n_kv; i++) {
        GGUFKV *kv = &g->kv[i];
        if (rd_str(f, &kv->key) || rd_u32(f, &kv->type))
            goto bad;
        if (kv->type == GGUF_T_STRING) {
            if (rd_str(f, &kv->str))
                goto bad;
        } else if (kv->type == GGUF_T_ARRAY) {
            if (rd_u32(f, &kv->arr_type) || rd_u64(f, &kv->arr_n))
                goto bad;
            kv->arr_pos = ftell(f);
            if (skip_array(f, kv->arr_type, kv->arr_n))
                goto bad;
        } else if (rd_scalar(f, kv->type, kv)) {
            goto bad;
        }
        if (!strcmp(kv->key, "general.alignment") && kv->type != GGUF_T_STRING
            && kv->type != GGUF_T_ARRAY)
            g->alignment = (uint32_t)kv->v.u;
    }
    if (g->alignment == 0 || (g->alignment & (g->alignment - 1)))
        goto bad;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        GGUFTensor *t = &g->t[i];
        uint32_t nd;
        if (rd_str(f, &t->name) || rd_u32(f, &nd) || nd < 1 || nd > 4)
            goto bad;
        t->n_dims = (int)nd;
        int64_t nel = 1;
        for (int d = 0; d < 4; d++)
            t->ne[d] = 1;
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t v;
            if (rd_u64(f, &v))
                goto bad;
            t->ne[d] = (int64_t)v;
            nel *= (int64_t)v;
        }
        if (rd_u32(f, &t->type) || rd_u64(f, &t->off))
            goto bad;

        const size_t row = gguf_type_bytes(t->type, t->ne[0]);
        t->nbytes = row ? (uint64_t)row * (uint64_t)(nel / t->ne[0]) : 0;
    }
    {
        const long pos = ftell(f);
        if (pos < 0)
            goto bad;
        const uint64_t a = g->alignment;
        g->data_off = ((uint64_t)pos + a - 1) / a * a;
    }
    return 0;
bad:
    snprintf(err, errlen, "%s: corrupt GGUF or unsupported type", path);
    gguf_close(g);
    return -1;
}

void gguf_close(GGUF *g)
{
    if (g->kv)
        for (uint64_t i = 0; i < g->n_kv; i++) {
            free(g->kv[i].key);
            free(g->kv[i].str);
        }
    if (g->t)
        for (uint64_t i = 0; i < g->n_tensors; i++)
            free(g->t[i].name);
    free(g->kv);
    free(g->t);
    if (g->fp)
        fclose(g->fp);
    memset(g, 0, sizeof(*g));
}

const GGUFKV *gguf_kv(const GGUF *g, const char *key)
{
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key && !strcmp(g->kv[i].key, key))
            return &g->kv[i];
    return NULL;
}

int gguf_u32(const GGUF *g, const char *key, uint32_t *out)
{
    const GGUFKV *kv = gguf_kv(g, key);
    if (!kv || kv->type == GGUF_T_STRING || kv->type == GGUF_T_ARRAY)
        return -1;
    if (kv->type == GGUF_T_FLOAT32 || kv->type == GGUF_T_FLOAT64)
        *out = (uint32_t)kv->v.f;
    else
        *out = (uint32_t)kv->v.u;
    return 0;
}

int gguf_f32(const GGUF *g, const char *key, float *out)
{
    const GGUFKV *kv = gguf_kv(g, key);
    if (!kv || kv->type == GGUF_T_STRING || kv->type == GGUF_T_ARRAY)
        return -1;
    if (kv->type == GGUF_T_FLOAT32 || kv->type == GGUF_T_FLOAT64)
        *out = (float)kv->v.f;
    else if (kv->type == GGUF_T_INT8 || kv->type == GGUF_T_INT16
             || kv->type == GGUF_T_INT32 || kv->type == GGUF_T_INT64)
        *out = (float)kv->v.i;
    else
        *out = (float)kv->v.u;
    return 0;
}

const char *gguf_str(const GGUF *g, const char *key)
{
    const GGUFKV *kv = gguf_kv(g, key);
    return kv && kv->type == GGUF_T_STRING ? kv->str : NULL;
}

int gguf_arr_i32(const GGUF *g, const char *key, int32_t *out, int max)
{
    const GGUFKV *kv = gguf_kv(g, key);
    if (!kv || kv->type != GGUF_T_ARRAY)
        return -1;
    const size_t s = scalar_size(kv->arr_type);
    if (!s || kv->arr_type == GGUF_T_FLOAT32 || kv->arr_type == GGUF_T_FLOAT64)
        return -1;
    const int n = kv->arr_n < (uint64_t)max ? (int)kv->arr_n : max;
    if (fseek(g->fp, kv->arr_pos, SEEK_SET))
        return -1;
    for (int i = 0; i < n; i++) {
        GGUFKV tmp;
        if (rd_scalar(g->fp, kv->arr_type, &tmp))
            return -1;
        out[i] = (int32_t)tmp.v.i;
        if (kv->arr_type == GGUF_T_UINT8 || kv->arr_type == GGUF_T_UINT16
            || kv->arr_type == GGUF_T_UINT32 || kv->arr_type == GGUF_T_UINT64
            || kv->arr_type == GGUF_T_BOOL)
            out[i] = (int32_t)tmp.v.u;
    }
    return n;
}

int gguf_arr_str(const GGUF *g, const char *key, char ***out)
{
    const GGUFKV *kv = gguf_kv(g, key);
    if (!kv || kv->type != GGUF_T_ARRAY || kv->arr_type != GGUF_T_STRING
        || kv->arr_n > (1u << 24) || fseek(g->fp, kv->arr_pos, SEEK_SET))
        return -1;
    const int n = (int)kv->arr_n;
    char **v = (char **)calloc((size_t)n + 1, sizeof(char *));
    if (!v)
        return -1;
    for (int i = 0; i < n; i++)
        if (rd_str(g->fp, &v[i])) {
            for (int j = 0; j < i; j++)
                free(v[j]);
            free(v);
            return -1;
        }
    *out = v;
    return n;
}

const GGUFTensor *gguf_tensor(const GGUF *g, const char *name)
{
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->t[i].name, name))
            return &g->t[i];
    return NULL;
}

int gguf_read(const GGUF *g, const GGUFTensor *t, uint64_t from, void *dst,
              size_t nbytes)
{
    if (from + nbytes > t->nbytes)
        return -1;
    const uint64_t off = g->data_off + t->off + from;
    if (fseek(g->fp, (long)off, SEEK_SET))
        return -1;
    return fread(dst, 1, nbytes, g->fp) == nbytes ? 0 : -1;
}
