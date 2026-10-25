/* Option parsing, vocab, timing and reports. */
#include "cli.h"

static int g_stop_ids[4], g_nstop;

double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int parse_ids(const char *s, int **out)
{
    int n = 0, cap = 64;
    int *v = (int *)malloc((size_t)cap * sizeof(int));
    while (v && *s) {
        char *end;
        const long x = strtol(s, &end, 10);
        if (end == s)
            break;
        if (n == cap) {
            cap *= 2;
            int *nv = (int *)realloc(v, (size_t)cap * sizeof(int));
            if (!nv) {
                free(v);
                return -1;
            }
            v = nv;
        }
        v[n++] = (int)x;
        s = end;
        while (*s == ',' || *s == ' ')
            s++;
    }
    *out = v;
    return v ? n : -1;
}

static char **g_vocab;
static int g_nvocab;

void load_vocab(const char *path)
{
    GGUF g;
    char err[256];
    if (gguf_open(&g, path, err, sizeof(err)))
        return;
    g_nvocab = gguf_arr_str(&g, "tokenizer.ggml.tokens", &g_vocab);
    gguf_close(&g);
}

static int cp_to_byte(unsigned cp)
{
    static int inv[324];
    static int ready;
    if (!ready) {
        int n = 0;
        for (int i = 0; i < 324; i++)
            inv[i] = -1;
        for (int b = 0; b < 256; b++) {
            const int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172)
                             || (b >= 174 && b <= 255);
            if (keep)
                inv[b] = b;
            else
                inv[256 + n++] = b;
        }
        ready = 1;
    }
    return cp < 324 ? inv[cp] : -1;
}

void print_token(FILE *f, int id)
{
    if (!g_vocab || id < 0 || id >= g_nvocab) {
        fprintf(f, "[%d]", id);
        return;
    }
    const unsigned char *p = (const unsigned char *)g_vocab[id];
    if (p[0] == '<' && p[1] == '|') {
        fputs((const char *)p, f);
        return;
    }
    while (*p) {
        unsigned cp;
        int len;
        if (*p < 0x80) { cp = *p; len = 1; }
        else if ((*p & 0xe0) == 0xc0) { cp = *p & 0x1f; len = 2; }
        else if ((*p & 0xf0) == 0xe0) { cp = *p & 0x0f; len = 3; }
        else { cp = *p & 0x07; len = 4; }
        for (int i = 1; i < len && p[i]; i++)
            cp = cp << 6 | (p[i] & 0x3f);
        const int b = cp_to_byte(cp);
        if (b >= 0)
            fputc(b, f);
        else
            fwrite(p, 1, (size_t)len, f);
        p += len;
    }
}

const char *g_model_path;

void load_vocab_once(const void *o)
{
    (void)o;
    if (!g_vocab && g_model_path)
        load_vocab(g_model_path);
}

static void top_k(const float *lg, int V, int K, int *id, double *lp)
{
    double mx = -INFINITY, sum = 0.0;
    for (int i = 0; i < V; i++)
        if (lg[i] > mx)
            mx = lg[i];
    for (int i = 0; i < V; i++)
        sum += exp((double)lg[i] - mx);
    const double lse = mx + log(sum);
    for (int k = 0; k < K; k++)
        id[k] = -1;
    for (int i = 0; i < V; i++) {
        int k = K;
        while (k > 0 && (id[k - 1] < 0 || lg[i] > lg[id[k - 1]]))
            k--;
        if (k < K) {
            memmove(id + k + 1, id + k, (size_t)(K - k - 1) * sizeof(int));
            id[k] = i;
        }
    }
    for (int k = 0; k < K; k++)
        lp[k] = (double)lg[id[k]] - lse;
}

void json_top(FILE *f, int pos, int chosen, const float *lg, int V, int K)
{
    int id[64];
    double lp[64];
    top_k(lg, V, K, id, lp);
    fprintf(f, "{\"pos\": %d, \"id\": %d, \"top\": [", pos, chosen);
    for (int k = 0; k < K; k++)
        fprintf(f, "%s[%d, %.6f]", k ? ", " : "", id[k], lp[k]);
    fprintf(f, "]}\n");
}

int argmax(const float *lg, int V)
{
    int b = 0;
    for (int i = 1; i < V; i++)
        if (lg[i] > lg[b])
            b = i;
    return b;
}

void report(const Q35Model *m, const Q35State *s, double secs, int ntok)
{
    static const char *kn[Q35_K_N] = {"embd", "head", "attention", "GDN",
                                      "GDN f32", "FFN"};
    uint64_t tot = 0;
    for (int k = 0; k < Q35_K_N; k++)
        tot += s->bytes_kind[k];
    fprintf(stderr, "%d tokens in %.2f s (%.3f s/token); weights read %.1f MB, "
            "%.2f GMAC\n", ntok, secs, ntok ? secs / ntok : 0.0, (double)tot / 1e6,
            (double)s->macs / 1e9);
    for (int k = 0; k < Q35_K_N; k++)
        if (s->bytes_kind[k])
            fprintf(stderr, "  %-9s %9.1f MB\n", kn[k], (double)s->bytes_kind[k] / 1e6);
    (void)m;
}

void load_stops(const char *path)
{
    GGUF g;
    char err[256];
    g_nstop = 0;
    if (gguf_open(&g, path, err, sizeof(err)))
        return;
    static const char *keys[] = {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                 "tokenizer.ggml.eom_token_id"};
    for (int i = 0; i < 3; i++) {
        uint32_t v;
        if (!gguf_u32(&g, keys[i], &v))
            g_stop_ids[g_nstop++] = (int)v;
    }
    gguf_close(&g);
}

int is_stop(int id)
{
    for (int i = 0; i < g_nstop; i++)
        if (g_stop_ids[i] == id)
            return 1;
    return 0;
}

int parse_opts(int argc, char **argv, int i0, Opts *o)
{
    o->ids = NULL;
    o->trace = NULL;
    o->gen = 0;
    o->chunk = 16;
    o->top = 10;
    o->ctx = 4096;
    o->kv_f16 = 1;
    o->print = 0;
    o->from = 1;
    o->state = Q35_STATE_F32;
    o->gamma = 8;
    o->ngram = 3;
    o->check = 0;
    o->stop = 0;
    o->stream = 0;
    o->temp = 0.0f;
    o->topp = 1.0f;
    o->topk = 0;
    o->presence = 0.0f;
    o->seed = 1;
    o->oracle = NULL;
    o->corrupt = 0.0;
    o->drafter = DR_MIX;
    o->tree = 0;
    o->keyed = 1;
    o->adj = 0;
    o->adj_file = NULL;
    o->adj_ro = 0;
    o->ro_rank = 16;
    o->tol = 0.0f;
    for (int i = i0; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--print")) { o->print = 1; continue; }
        if (!strcmp(a, "--check")) { o->check = 1; continue; }
        if (!strcmp(a, "--stop")) { o->stop = 1; continue; }
        if (!strcmp(a, "--stream")) { o->stream = 1; continue; }
        if (!v) { fprintf(stderr, "missing value for %s\n", a); return -1; }
        if (!strcmp(a, "--ids")) o->ids = v;
        else if (!strcmp(a, "--gen")) o->gen = atoi(v);
        else if (!strcmp(a, "--chunk")) o->chunk = atoi(v);
        else if (!strcmp(a, "--top")) o->top = atoi(v);
        else if (!strcmp(a, "--ctx")) o->ctx = atoi(v);
        else if (!strcmp(a, "--from")) o->from = atoi(v);
        else if (!strcmp(a, "--trace")) o->trace = v;
        else if (!strcmp(a, "--gamma")) o->gamma = atoi(v);
        else if (!strcmp(a, "--ngram")) o->ngram = atoi(v);
        else if (!strcmp(a, "--oracle")) o->oracle = v;
        else if (!strcmp(a, "--temp")) o->temp = (float)atof(v);
        else if (!strcmp(a, "--top-p")) o->topp = (float)atof(v);
        else if (!strcmp(a, "--top-k")) o->topk = atoi(v);
        else if (!strcmp(a, "--presence")) o->presence = (float)atof(v);
        else if (!strcmp(a, "--seed")) o->seed = strtoull(v, NULL, 10);
        else if (!strcmp(a, "--corrupt")) o->corrupt = atof(v);
        else if (!strcmp(a, "--tree")) o->tree = atoi(v);
        else if (!strcmp(a, "--keyed")) o->keyed = atoi(v) != 0;
        else if (!strcmp(a, "--adj")) o->adj = atoi(v) != 0;
        else if (!strcmp(a, "--adj-file")) o->adj_file = v;
        else if (!strcmp(a, "--adj-ro")) o->adj_ro = atoi(v) != 0;
        else if (!strcmp(a, "--ro-rank")) o->ro_rank = atoi(v);
        else if (!strcmp(a, "--tol")) o->tol = (float)atof(v);
        else if (!strcmp(a, "--drafter")) {
            if (!strcmp(v, "pld")) o->drafter = DR_PLD;
            else if (!strcmp(v, "jacobi")) o->drafter = DR_JACOBI;
            else if (!strcmp(v, "mix")) o->drafter = DR_MIX;
            else if (!strcmp(v, "none")) o->drafter = DR_NONE;
            else { fprintf(stderr, "--drafter pld|jacobi|mix|none\n"); return -1; }
        }
        else if (!strcmp(a, "--kv")) o->kv_f16 = strcmp(v, "f32") != 0;
        else if (!strcmp(a, "--state")) {
            if (!strcmp(v, "f32")) o->state = Q35_STATE_F32;
            else if (!strcmp(v, "f16")) o->state = Q35_STATE_F16;
            else if (!strcmp(v, "bf16")) o->state = Q35_STATE_BF16;
            else if (!strcmp(v, "int8")) o->state = Q35_STATE_I8;
            else if (!strcmp(v, "ro32")) o->state = Q35_STATE_RO32;
            else if (!strcmp(v, "ro8")) o->state = Q35_STATE_RO8;
            else if (!strcmp(v, "ro16")) o->state = Q35_STATE_RO16;
            else { fprintf(stderr, "--state f32|f16|bf16|int8|ro32|ro16|ro8\n"); return -1; }
        }
        else { fprintf(stderr, "unknown option %s\n", a); return -1; }
        i++;
    }
    if (o->chunk < 1) o->chunk = 1;
    if (o->top < 1) o->top = 1;
    if (o->top > 64) o->top = 64;
    return 0;
}
