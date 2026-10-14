// Dump llama.cpp logits in the Q35R format used by q35 (reference for parity tests).
//
//   llama_logits model.gguf out.bin (--ids 1,2,3 | --text "...") [--gen N]
//                [--fa on|off] [--kv f16|f32] [--threads N] [--ubatch N]
//                [--tokens-only 1] [--dump prefix:file]
//
// Build against llama.cpp (L = its directory):
//   g++ -O2 -std=c++17 llama_logits.cpp -I$L/include -I$L/ggml/include \
//       -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o llama_logits
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"

// record: name[64], ne0, ne1 (u32), f32 data
static FILE *g_dump;
static std::string g_prefix;

static bool on_tensor(struct ggml_tensor *t, bool ask, void *)
{
    const bool want = g_dump && !strncmp(t->name, g_prefix.c_str(), g_prefix.size())
                      && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    if (ask)
        return want;
    if (want) {
        char name[64] = {0};
        strncpy(name, t->name, 63);
        const uint32_t ne[2] = {(uint32_t)t->ne[0], (uint32_t)(ggml_nelements(t) / t->ne[0])};
        fwrite(name, 1, 64, g_dump);
        fwrite(ne, 4, 2, g_dump);
        fwrite(t->data, 4, (size_t)ggml_nelements(t), g_dump);
    }
    return true;
}

static void quiet(ggml_log_level level, const char *text, void *)
{
    if (level >= GGML_LOG_LEVEL_WARN)
        fputs(text, stderr);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf out.bin (--ids 1,2,3 | --text \"...\") [--gen N] "
                        "[--fa on|off] [--kv f16|f32] [--threads N] [--ubatch N]\n", argv[0]);
        return 2;
    }
    std::vector<llama_token> ids;
    const char *ids_s = nullptr, *text = nullptr;
    bool fa = true, kv16 = true;
    int threads = 4, ubatch = 512, gen = 0;
    bool tokens_only = false;
    for (int i = 3; i + 1 < argc; i += 2) {
        const char *a = argv[i], *v = argv[i + 1];
        if (!strcmp(a, "--ids")) ids_s = v;
        else if (!strcmp(a, "--text")) text = v;
        else if (!strcmp(a, "--gen")) gen = atoi(v);
        else if (!strcmp(a, "--tokens-only")) tokens_only = strcmp(v, "0") != 0;
        else if (!strcmp(a, "--fa")) fa = strcmp(v, "off") != 0;
        else if (!strcmp(a, "--kv")) kv16 = strcmp(v, "f32") != 0;
        else if (!strcmp(a, "--threads")) threads = atoi(v);
        else if (!strcmp(a, "--ubatch")) ubatch = atoi(v);
        else if (!strcmp(a, "--dump")) {
            const char *colon = strchr(v, ':');
            if (!colon) { fprintf(stderr, "--dump prefix:file\n"); return 2; }
            g_prefix.assign(v, colon - v);
            g_dump = fopen(colon + 1, "wb");
        }
        else { fprintf(stderr, "unknown option %s\n", a); return 2; }
    }
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int V = llama_vocab_n_tokens(vocab);
    if (ids_s) {
        for (const char *p = ids_s; *p;) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p)
                break;
            ids.push_back((llama_token)v);
            p = end;
            while (*p == ',' || *p == ' ')
                p++;
        }
    } else if (text) {
        const int len = (int)strlen(text);
        ids.resize((size_t)len + 16);
        const int n = llama_tokenize(vocab, text, len, ids.data(), (int)ids.size(), true, true);
        if (n < 0) {
            fprintf(stderr, "tokenize failed\n");
            return 1;
        }
        ids.resize((size_t)n);
    }
    if (ids.empty()) {
        fprintf(stderr, "empty sequence\n");
        return 1;
    }
    if (tokens_only) {
        FILE *f = fopen(argv[2], "wb");
        if (!f)
            return 1;
        const uint32_t h[3] = {(uint32_t)ids.size(), 0u, 0u};
        fwrite("Q35R", 1, 4, f);
        fwrite(h, 4, 3, f);
        for (const llama_token t : ids) {
            const int32_t v = t;
            fwrite(&v, 4, 1, f);
        }
        fclose(f);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    const int n0 = (int)ids.size(), ntot = n0 + gen;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(ntot < 256 ? 256 : ntot + 16);
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = (uint32_t)ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = kv16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    if (g_dump) {
        cp.cb_eval = on_tensor;
        cp.cb_eval_user_data = nullptr;
    }
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create context\n");
        return 1;
    }
    std::vector<float> all((size_t)ntot * V);
    llama_batch b = llama_batch_init(n0 > 1 ? n0 : 1, 0, 1);
    for (int i = 0; i < n0; i++) {
        b.token[i] = ids[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n0;
    if (llama_decode(ctx, b)) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }
    for (int i = 0; i < n0; i++)
        memcpy(&all[(size_t)i * V], llama_get_logits_ith(ctx, i), sizeof(float) * V);
    for (int g = 0; g < gen; g++) {
        const float *lg = &all[(size_t)(n0 + g - 1) * V];
        int best = 0;
        for (int k = 1; k < V; k++)
            if (lg[k] > lg[best])
                best = k;
        ids.push_back(best);
        b.token[0] = best;
        b.pos[0] = n0 + g;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        b.n_tokens = 1;
        if (llama_decode(ctx, b)) {
            fprintf(stderr, "llama_decode failed\n");
            return 1;
        }
        memcpy(&all[(size_t)(n0 + g) * V], llama_get_logits_ith(ctx, 0), sizeof(float) * V);
    }
    FILE *f = fopen(argv[2], "wb");
    if (!f)
        return 1;
    const uint32_t h[3] = {(uint32_t)ntot, (uint32_t)V, kv16 ? 1u : 0u};
    fwrite("Q35R", 1, 4, f);
    fwrite(h, 4, 3, f);
    for (int i = 0; i < ntot; i++) {
        const int32_t t = ids[i];
        fwrite(&t, 4, 1, f);
    }
    fwrite(all.data(), sizeof(float), all.size(), f);
    fclose(f);
    if (g_dump)
        fclose(g_dump);
    // decoded text between <<< and >>>
    if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE)
        fputs("\n<<<\n", stderr);
    for (int i = 0; llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE && i < ntot; i++) {
        char buf[256];
        const int k = llama_token_to_piece(vocab, ids[i], buf, sizeof(buf), 0, true);
        if (k > 0)
            fwrite(buf, 1, (size_t)k, stderr);
        if (i == n0 - 1)
            fputs("[[", stderr);
    }
    if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE)
        fputs("]]\n>>>\n", stderr);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
