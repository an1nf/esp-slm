/* run, score, test, dump, info and meta. */
#include "cli.h"

int cmd_run(const Q35Model *m, Q35State *s, const Opts *o, int score)
{
    int *ids;
    const int n = o->ids ? parse_ids(o->ids, &ids) : -1;
    if (n < 1) {
        fprintf(stderr, "--ids required\n");
        return 1;
    }
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= m->c.vocab) {
            fprintf(stderr, "token id out of range: %d\n", ids[i]);
            return 1;
        }
    const int V = m->c.vocab;
    if (n + o->gen > s->max_seq) {
        fprintf(stderr, "context too short (--ctx)\n");
        return 1;
    }
    if (o->print)
        for (int i = 0; i < n; i++)
            print_token(stderr, ids[i]);
    const double t0 = now_s();
    int done = 0;
    const float *lg = NULL;
    while (done < n) {
        const int T = n - done < o->chunk ? n - done : o->chunk;
        lg = q35_forward(m, s, ids + done, T, score);
        if (!lg) {
            fprintf(stderr, "q35_forward failed\n");
            return 1;
        }
        if (score)
            for (int t = 0; t < T; t++) {
                const int p = done + t;
                if (p + 1 >= o->from && p + 1 < n)
                    json_top(stdout, p + 1, ids[p + 1], lg + (size_t)t * V, V, o->top);
            }
        done += T;
    }
    const double t1 = now_s();
    fprintf(stderr, "\nprefill: ");
    report(m, s, t1 - t0, n);
    if (score)
        return 0;
    memset(s->bytes_kind, 0, sizeof(s->bytes_kind));
    s->macs = 0;
    for (int g = 0; g < o->gen; g++) {
        const int next = argmax(lg, V);
        json_top(stdout, n + g, next, lg, V, o->top);
        fflush(stdout);
        if (o->print) {
            print_token(stderr, next);
            fflush(stderr);
        }
        if (g + 1 == o->gen)
            break;
        lg = q35_forward(m, s, &next, 1, 0);
        if (!lg)
            return 1;
    }
    const double t2 = now_s();
    if (o->gen > 1) {
        fprintf(stderr, "\ngenerated: ");
        report(m, s, t2 - t1, o->gen - 1);
    }
    free(ids);
    return 0;
}

int cmd_test(const Q35Model *m, const char *ref)
{
    FILE *f = fopen(ref, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", ref);
        return 1;
    }
    char magic[4];
    uint32_t h[3];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "Q35R", 4)
        || fread(h, 4, 3, f) != 3 || (int)h[1] != m->c.vocab) {
        fprintf(stderr, "%s: invalid reference\n", ref);
        fclose(f);
        return 1;
    }
    const int n = (int)h[0], V = (int)h[1], kv_f16 = (int)h[2];
    int *ids = (int *)malloc((size_t)n * sizeof(int));
    float *want = (float *)malloc((size_t)n * V * sizeof(float));
    if (!ids || !want || fread(ids, 4, (size_t)n, f) != (size_t)n
        || fread(want, 4, (size_t)n * V, f) != (size_t)n * V) {
        fprintf(stderr, "%s: truncated\n", ref);
        fclose(f);
        return 1;
    }
    fclose(f);
    int fails = 0;
    static const int chunks[][4] = {{1, 0, 0, 0}, {3, 2, 64, 0}, {64, 0, 0, 0}};
    for (int c = 0; c < 3; c++) {
        Q35State s;
        if (q35_state_init(&s, m, n + 1, 64, kv_f16)) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        long diff = 0;
        double maxd = 0.0;
        int done = 0, ci = 0;
        while (done < n) {
            int T = chunks[c][ci] ? chunks[c][ci] : chunks[c][ci - 1];
            if (chunks[c][ci])
                ci++;
            if (T > n - done)
                T = n - done;
            const float *lg = q35_forward(m, &s, ids + done, T, 1);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < V; i++) {
                    const float a = lg[(size_t)t * V + i];
                    const float b = want[(size_t)(done + t) * V + i];
                    if (memcmp(&a, &b, 4)) {
                        diff++;
                        const double d = fabs((double)a - (double)b);
                        if (d > maxd)
                            maxd = d;
                    }
                }
            done += T;
        }
        printf("  chunks %d,%d,...: %ld of %ld logits differ (max diff %.3g)\n",
               chunks[c][0], chunks[c][1], diff, (long)n * V, maxd);
        fails += diff != 0;
        q35_state_free(&s);
    }
    free(ids);
    free(want);
    printf("%s\n", fails ? "FAIL" : "OK: bit-exact");
    return fails ? 1 : 0;
}

int cmd_dump(const Q35Model *m, Q35State *s, const Opts *o, const char *out)
{
    int *ids;
    const int n = o->ids ? parse_ids(o->ids, &ids) : -1;
    if (n < 1 || n > s->max_seq) {
        fprintf(stderr, "--ids required (and enough --ctx)\n");
        return 1;
    }
    FILE *f = fopen(out, "wb");
    if (!f)
        return 1;
    const uint32_t h[3] = {(uint32_t)n, (uint32_t)m->c.vocab, (uint32_t)s->kv_f16};
    fwrite("Q35R", 1, 4, f);
    fwrite(h, 4, 3, f);
    fwrite(ids, 4, (size_t)n, f);
    int done = 0;
    while (done < n) {
        const int T = n - done < o->chunk ? n - done : o->chunk;
        const float *lg = q35_forward(m, s, ids + done, T, 1);
        if (!lg) {
            fclose(f);
            return 1;
        }
        fwrite(lg, 4, (size_t)T * m->c.vocab, f);
        done += T;
    }
    fclose(f);
    free(ids);
    return 0;
}

int cmd_meta(const char *path)
{
    GGUF g;
    char err[256];
    if (gguf_open(&g, path, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    printf("GGUF v%u, %llu keys, %llu tensors, alignment %u\n", g.version,
           (unsigned long long)g.n_kv, (unsigned long long)g.n_tensors, g.alignment);
    for (uint64_t i = 0; i < g.n_kv; i++) {
        const GGUFKV *kv = &g.kv[i];
        printf("  %s = ", kv->key);
        if (kv->type == GGUF_T_STRING)
            printf("\"%.120s\"\n", kv->str);
        else if (kv->type == GGUF_T_ARRAY) {
            printf("[array type %u, %llu items]", kv->arr_type, (unsigned long long)kv->arr_n);
            if (kv->arr_type == GGUF_T_STRING && kv->arr_n <= 64 && strncmp(kv->key, "tokenizer.", 10)) {
                char **v;
                const int n = gguf_arr_str(&g, kv->key, &v);
                for (int j = 0; j < n; j++) {
                    printf("%s%s", j ? ", " : " ", v[j]);
                    free(v[j]);
                }
                if (n >= 0)
                    free(v);
            } else if (kv->arr_type != GGUF_T_STRING && kv->arr_type != GGUF_T_FLOAT32
                       && kv->arr_type != GGUF_T_FLOAT64 && kv->arr_n <= 16) {
                int32_t v[16];
                const int n = gguf_arr_i32(&g, kv->key, v, 16);
                for (int j = 0; j < n; j++)
                    printf("%s%d", j ? ", " : " ", v[j]);
            }
            printf("\n");
        } else if (kv->type == GGUF_T_FLOAT32 || kv->type == GGUF_T_FLOAT64)
            printf("%g\n", kv->v.f);
        else if (kv->type == GGUF_T_INT8 || kv->type == GGUF_T_INT16 || kv->type == GGUF_T_INT32
                 || kv->type == GGUF_T_INT64)
            printf("%lld\n", (long long)kv->v.i);
        else
            printf("%llu\n", (unsigned long long)kv->v.u);
    }

    uint64_t cnt[256] = {0}, by[256] = {0};
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const uint32_t t = g.t[i].type < 256 ? g.t[i].type : 255;
        cnt[t]++;
        by[t] += g.t[i].nbytes;
    }
    for (int t = 0; t < 256; t++)
        if (cnt[t])
            printf("  tensors %-7s (%3d): %4llu, %.1f MB\n", gguf_type_name((uint32_t)t), t,
                   (unsigned long long)cnt[t], (double)by[t] / 1e6);
    for (uint64_t i = 0; i < g.n_tensors && i < 40; i++)
        printf("    %-40s %-7s [%lld, %lld]\n", g.t[i].name, gguf_type_name(g.t[i].type),
               (long long)g.t[i].ne[0], (long long)g.t[i].ne[1]);
    gguf_close(&g);
    return 0;
}
