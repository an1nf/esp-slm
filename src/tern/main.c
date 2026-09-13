/* tern: selftest, bench, run, test, lanes. */
#include "tern.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t rng_state = 0x12345678u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int argmax(const float *x, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b])
            b = i;
    return b;
}

static int cmd_selftest(void)
{
    tern_lut_init();
    int fails = 0;
    const int shapes[][2] = {{1, 1}, {3, 4}, {5, 5}, {7, 13}, {16, 64},
                             {33, 96}, {64, 2560}, {17, 6912}, {9, 1001}};
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        const int rows = shapes[si][0], cols = shapes[si][1];
        int8_t *w = malloc((size_t)rows * cols), *x = malloc((size_t)cols);
        int8_t *tile = malloc((size_t)tern_tile_bytes(cols));
        int32_t *ref = malloc(sizeof(int32_t) * rows);
        int32_t *got = malloc(sizeof(int32_t) * rows);
        int32_t *got2 = malloc(sizeof(int32_t) * rows);
        for (int i = 0; i < rows * cols; i++)
            w[i] = (int8_t)((int)(rng() % 3) - 1);
        for (int i = 0; i < cols; i++)
            x[i] = (int8_t)((int)(rng() % 256) - 128);
        for (int r = 0; r < rows; r++) {
            int32_t a = 0;
            for (int c = 0; c < cols; c++)
                a += (int32_t)w[(size_t)r * cols + c] * x[c];
            ref[r] = a;
        }
        for (int fmt = TERN_FMT_T2; fmt <= TERN_FMT_T3; fmt++) {
            const int rb = tern_row_bytes(fmt, cols);
            uint8_t *pk = malloc((size_t)rows * rb);
            for (int r = 0; r < rows; r++)
                tern_pack_row(fmt, w + (size_t)r * cols, cols, pk + (size_t)r * rb);
            tern_gemv_rows(fmt, pk, rows, cols, rb, x, tile, got);
            tern_gemv_ref(fmt, pk, rows, cols, rb, x, got2);
            int bad = memcmp(ref, got, sizeof(int32_t) * rows) != 0
                   || memcmp(ref, got2, sizeof(int32_t) * rows) != 0;
            if (fmt == TERN_FMT_T3)
                for (int i = 0; i < rows * rb; i++)
                    bad |= pk[i] >= 243;

            tern_decode_row(fmt, pk, cols, tile);
            bad |= memcmp(tile, w, (size_t)cols) != 0;
            printf("  gemv %-2s %5dx%-5d %s\n", fmt == TERN_FMT_T2 ? "T2" : "T3",
                   rows, cols, bad ? "FAIL" : "ok");
            fails += bad;
            free(pk);
        }
        free(w); free(x); free(tile); free(ref); free(got); free(got2);
    }

    {
        float xs[6] = {0.0f, 1.0f, -1.0f, 0.5f / 127.0f, -2.0f, 2.0f};
        int8_t q[6];
        float s = tern_quant_act(xs, 6, q);
        int bad = !(q[0] == 0 && q[4] == -127 && q[5] == 127 && s == 63.5f);
        printf("  quant_act edge cases %s\n", bad ? "FAIL" : "ok");
        fails += bad;
    }
    printf(fails ? "SELFTEST: FAIL (%d)\n" : "SELFTEST: OK\n", fails);
    return fails ? 1 : 0;
}

static int cmd_bench(int rows, int cols, int reps)
{
    tern_lut_init();
    int8_t *w = malloc((size_t)rows * cols), *x = malloc((size_t)cols);
    int8_t *tile = malloc((size_t)tern_tile_bytes(cols));
    int32_t *acc = malloc(sizeof(int32_t) * rows);
    for (size_t i = 0; i < (size_t)rows * cols; i++)
        w[i] = (int8_t)((int)(rng() % 3) - 1);
    for (int i = 0; i < cols; i++)
        x[i] = (int8_t)((int)(rng() % 256) - 128);
    for (int fmt = TERN_FMT_T2; fmt <= TERN_FMT_T3; fmt++) {
        const int rb = tern_row_bytes(fmt, cols);
        uint8_t *pk = malloc((size_t)rows * rb);
        for (int r = 0; r < rows; r++)
            tern_pack_row(fmt, w + (size_t)r * cols, cols, pk + (size_t)r * rb);
        for (int variant = 0; variant < 2; variant++) {
            if (variant == 0)
                tern_gemv_rows(fmt, pk, rows, cols, rb, x, tile, acc);
            else
                tern_gemv_ref(fmt, pk, rows, cols, rb, x, acc);
            double t0 = (double)clock() / CLOCKS_PER_SEC;
            for (int i = 0; i < reps; i++) {
                if (variant == 0)
                    tern_gemv_rows(fmt, pk, rows, cols, rb, x, tile, acc);
                else
                    tern_gemv_ref(fmt, pk, rows, cols, rb, x, acc);
            }
            double dt = (double)clock() / CLOCKS_PER_SEC - t0;
            double macs = (double)rows * cols * reps;
            printf("  gemv %s %dx%d %-17s %.2f GMAC/s, %.0f MB/s weights\n",
                   fmt == TERN_FMT_T2 ? "T2" : "T3", rows, cols,
                   variant == 0 ? "tile+product:" : "direct table:",
                   macs / dt / 1e9, (double)rows * rb * reps / dt / 1e6);
        }
        free(pk);
    }
    free(w); free(x); free(tile); free(acc);
    return 0;
}

typedef struct {
    const char *place, *trace, *logits;
    int kv_fmt;
    size_t buf, chunk;
} Opts;

static int parse_common(Opts *o, int argc, char **argv, int i)
{
    if (i + 1 >= argc)
        return 0;
    if (strcmp(argv[i], "--place") == 0) {
        o->place = argv[i + 1];
    } else if (strcmp(argv[i], "--trace") == 0) {
        o->trace = argv[i + 1];
    } else if (strcmp(argv[i], "--buf") == 0) {
        o->buf = (size_t)atol(argv[i + 1]) * 1024u;
    } else if (strcmp(argv[i], "--chunk") == 0) {
        o->chunk = (size_t)atol(argv[i + 1]) * 1024u;
    } else if (strcmp(argv[i], "--kv") == 0) {
        if (strcmp(argv[i + 1], "int8") == 0)
            o->kv_fmt = TERN_KV_I8;
        else if (strcmp(argv[i + 1], "f32") == 0)
            o->kv_fmt = TERN_KV_F32;
        else
            return 0;
    } else {
        return 0;
    }
    return 2;
}

static int open_all(const Opts *o, const char *path, TernModel *m, TernState *s)
{
    char err[256];
    if (tern_open(m, path, o->place, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 2;
    }
    if (tern_state_init(s, m, o->kv_fmt, o->buf) != 0) {
        fprintf(stderr, "error: out of memory for state\n");
        tern_free(m);
        return 2;
    }
    if (o->chunk)
        tern_set_chunk(s, o->chunk);
    if (o->trace) {
        s->trace = fopen(o->trace, "w");
        if (!s->trace) {
            fprintf(stderr, "error: cannot write %s\n", o->trace);
            return 2;
        }
        fprintf(s->trace, "pos,kind,unit,r0,r1,bytes,macs\n");
    }
    return 0;
}

static void close_all(TernModel *m, TernState *s)
{
    if (s->trace)
        fclose(s->trace);
    tern_state_free(s);
    tern_free(m);
}

static int parse_ids(const char *s, int *ids, int max)
{
    int n = 0;
    while (*s && n < max) {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s)
            break;
        ids[n++] = (int)v;
        s = (*end == ',') ? end + 1 : end;
    }
    return n;
}

static double now_s(void)
{
    return (double)clock() / CLOCKS_PER_SEC;
}

static void print_placement(const TernModel *m, const TernState *s)
{
    printf("placement: PSRAM %.2f MB, flash %.2f MB resident; SD %.2f MB "
           "streamed (buffer %zu KiB); norms %.2f MB\n",
           m->placed[TERN_TIER_PSRAM] / 1e6, m->placed[TERN_TIER_FLASH] / 1e6,
           m->placed[TERN_TIER_SD] / 1e6, s->sbuf_bytes / 1024, m->aux_bytes / 1e6);
}

static int cmd_run(const Opts *o, const char *path, const char *ids_s, int gen)
{
    TernModel m;
    TernState s;
    int rc = open_all(o, path, &m, &s);
    if (rc)
        return rc;
    const TernConfig *c = &m.cfg;
    int *ids = malloc(sizeof(int) * (size_t)c->max_seq);
    int n = parse_ids(ids_s, ids, c->max_seq);
    if (n == 0) {
        fprintf(stderr, "error: empty --ids\n");
        return 2;
    }
    printf("model: dim=%d layers=%d heads=%d/%d inter=%d vocab=%d format=%s "
           "head=%s kv=%s\n",
           c->dim, c->n_layers, c->n_heads, c->n_kv, c->inter, c->vocab,
           c->fmt == TERN_FMT_T2 ? "T2" : "T3",
           c->head_fmt == TERN_HEAD_F32 ? "f32" : "int8",
           s.kv_fmt == TERN_KV_F32 ? "f32" : "int8");
    print_placement(&m, &s);
    int pos = 0, next = 0;
    const float *lg = NULL;
    double t0 = now_s();
    for (; pos < n; pos++)
        lg = tern_forward(&m, &s, ids[pos], pos);
    double t_prefill = now_s() - t0;
    next = argmax(lg, c->vocab);
    tern_state_reset_counters(&s);
    printf("generated:");
    t0 = now_s();
    int produced = 0;
    for (int g = 0; g < gen && pos < c->max_seq; g++) {
        printf(" %d", next);
        fflush(stdout);
        produced++;
        if (g == gen - 1)
            break;
        lg = tern_forward(&m, &s, next, pos++);
        next = argmax(lg, c->vocab);
    }
    double t_gen = now_s() - t0;
    int steps = produced > 1 ? produced - 1 : 1;
    printf("\nprefill: %d tokens in %.2f s\n", n, t_prefill);
    printf("decode: %.1f ms/token (host CPU, one thread)\n",
           1000.0 * t_gen / steps);
    printf("bytes per token: ternary %.3f MB, head %.3f MB, embedding %.4f MB, "
           "KV %.3f MB, norms %.3f MB; MAC/token %.2f M\n",
           s.bytes_ternary / 1e6 / steps, s.bytes_head / 1e6 / steps,
           s.bytes_emb / 1e6 / steps, s.bytes_kv / 1e6 / steps,
           s.bytes_aux / 1e6 / steps, s.macs / 1e6 / steps);
    printf("weights read per tier and token: PSRAM %.3f MB, flash %.3f MB, "
           "SD %.3f MB (%.0f reads)\n",
           s.tier_bytes[TERN_TIER_PSRAM] / 1e6 / steps,
           s.tier_bytes[TERN_TIER_FLASH] / 1e6 / steps,
           s.tier_bytes[TERN_TIER_SD] / 1e6 / steps,
           (double)s.tier_reads[TERN_TIER_SD] / steps);
    if (s.io_errors)
        printf("WARNING: %llu read errors\n", (unsigned long long)s.io_errors);
    free(ids);
    rc = s.io_errors ? 1 : 0;
    close_all(&m, &s);
    return rc;
}

static int cmd_test(const Opts *o, const char *mpath, const char *rpath, double tol)
{
    TernModel m;
    TernState s;
    int rc = open_all(o, mpath, &m, &s);
    if (rc)
        return rc;
    FILE *f = fopen(rpath, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", rpath);
        return 2;
    }
    char magic[4];
    uint32_t hdr[3];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TREF", 4) != 0
        || fread(hdr, 4, 3, f) != 3) {
        fprintf(stderr, "error: malformed reference\n");
        return 2;
    }
    const int vocab = (int)hdr[0], np = (int)hdr[1], ng = (int)hdr[2];
    if (vocab != m.cfg.vocab || np < 1 || ng < 1 || np + ng > m.cfg.max_seq) {
        fprintf(stderr, "error: reference does not match model\n");
        return 2;
    }
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(np + ng));
    const size_t nfed = (size_t)(np + ng - 1);
    float *ref = malloc(sizeof(float) * nfed * vocab);
    if (fread(ids, 4, (size_t)(np + ng), f) != (size_t)(np + ng)
        || fread(ref, 4, nfed * vocab, f) != nfed * vocab) {
        fprintf(stderr, "error: truncated reference\n");
        return 2;
    }
    fclose(f);
    FILE *lf = o->logits ? fopen(o->logits, "wb") : NULL;
    if (o->logits && !lf) {
        fprintf(stderr, "error: cannot write %s\n", o->logits);
        return 2;
    }

    double max_abs = 0.0, max_ref = 0.0;
    int agree = 0, checked = 0;
    for (size_t p = 0; p < nfed; p++) {
        const float *lg = tern_forward(&m, &s, ids[p], (int)p);
        if (lf)
            fwrite(lg, sizeof(float), (size_t)vocab, lf);
        const float *r = ref + p * vocab;
        for (int v = 0; v < vocab; v++) {
            double d = fabs((double)lg[v] - (double)r[v]);
            if (d > max_abs)
                max_abs = d;
            if (fabs(r[v]) > max_ref)
                max_ref = fabs(r[v]);
        }
        if (p + 1 >= (size_t)np) {
            checked++;
            agree += argmax(lg, vocab) == ids[p + 1];
        }
    }
    if (lf)
        fclose(lf);
    const double lim = tol * (max_ref > 1.0 ? max_ref : 1.0);
    const int pass = agree == checked && max_abs <= lim && s.io_errors == 0;
    printf("parity: tokens %d/%d equal, max dlogit %.3g (limit %.3g, "
           "max logit %.3g) -> %s\n",
           agree, checked, max_abs, lim, max_ref, pass ? "OK" : "FAIL");
    printf("weight bytes per tier (psram,flash,sd): %llu,%llu,%llu\n",
           (unsigned long long)s.tier_bytes[TERN_TIER_PSRAM],
           (unsigned long long)s.tier_bytes[TERN_TIER_FLASH],
           (unsigned long long)s.tier_bytes[TERN_TIER_SD]);
    printf("reads per tier (psram,flash,sd): %llu,%llu,%llu\n",
           (unsigned long long)s.tier_reads[TERN_TIER_PSRAM],
           (unsigned long long)s.tier_reads[TERN_TIER_FLASH],
           (unsigned long long)s.tier_reads[TERN_TIER_SD]);
    free(ids); free(ref);
    close_all(&m, &s);
    return pass ? 0 : 1;
}

static int usage(void)
{
    fprintf(stderr,
            "usage:\n"
            "  tern selftest\n"
            "  tern bench [rows cols reps]\n"
            "  tern run  <model.tbin> --ids 1,2,3 [--gen N] [options]\n"
            "  tern test <model.tbin> <ref.bin> [--tol X] [--logits out.bin] [options]\n"
            "  tern lanes <model.tbin> --lanes N --steps P [--bw psram=MBps,flash=..,sd=..]\n"
            "             [--lat sd=ms] [--ring KiB] [--chunk KiB] [--place f] [--sync] [--check]\n"
            "options: --place file  --kv f32|int8  --buf KiB  --chunk KiB  "
            "--trace file\n");
    return 2;
}

int cmd_lanes(int argc, char **argv);

int main(int argc, char **argv)
{
    Opts o = {NULL, NULL, NULL, TERN_KV_F32, 0, 0};
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0)
        return cmd_selftest();
    if (argc >= 3 && strcmp(argv[1], "lanes") == 0)
        return cmd_lanes(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "bench") == 0)
        return cmd_bench(argc > 2 ? atoi(argv[2]) : 6912,
                         argc > 3 ? atoi(argv[3]) : 2560,
                         argc > 4 ? atoi(argv[4]) : 20);
    if (argc >= 3 && strcmp(argv[1], "run") == 0) {
        const char *ids = NULL;
        int gen = 16;
        for (int i = 3; i < argc; i++) {
            int k = parse_common(&o, argc, argv, i);
            if (k) {
                i += k - 1;
            } else if (strcmp(argv[i], "--ids") == 0 && i + 1 < argc) {
                ids = argv[++i];
            } else if (strcmp(argv[i], "--gen") == 0 && i + 1 < argc) {
                gen = atoi(argv[++i]);
            } else {
                fprintf(stderr, "unknown option: %s\n", argv[i]);
                return usage();
            }
        }
        if (!ids)
            return usage();
        return cmd_run(&o, argv[2], ids, gen);
    }
    if (argc >= 4 && strcmp(argv[1], "test") == 0) {
        double tol = 1e-4;
        for (int i = 4; i < argc; i++) {
            int k = parse_common(&o, argc, argv, i);
            if (k) {
                i += k - 1;
            } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
                tol = atof(argv[++i]);
            } else if (strcmp(argv[i], "--logits") == 0 && i + 1 < argc) {
                o.logits = argv[++i];
            } else {
                fprintf(stderr, "unknown option: %s\n", argv[i]);
                return usage();
            }
        }
        return cmd_test(&o, argv[2], argv[3], tol);
    }
    return usage();
}
