/* q35 command dispatch. */
#include "cli.h"

static void usage(void)
{
    fprintf(stderr,
            "usage: q35 info <gguf>\n"
            "     q35 meta  <gguf>            (metadata and tensor types)\n"
            "     q35 run   <gguf> --ids 1,2,3 [--gen N] [--chunk T] [--top K] [--ctx N] [--kv f16|f32]\n"
            "                      [--state f32|f16|bf16|int8] [--print] [--trace f]\n"
            "     q35 score <gguf> --ids ... --from P [--chunk T] [--top K] [--ctx N] [--kv f16|f32]\n"
            "     q35 test  <gguf> <ref.bin>\n"
            "     q35 treetest <gguf> --ids ... [--gamma N] [--seed S] [--kv ...] [--state ...] [--tol x]\n"
            "     q35 dump  <gguf> <out.bin> --ids ... [--chunk T] [--kv f16|f32] [--state ...]\n"
            "     q35 spec  <gguf> --ids ... --gen N [--gamma G] [--ngram K] [--drafter pld|jacobi|mix|none]\n"
            "                      [--check] [--print] [--stop] [--stream]\n"
            "                      [--temp T] [--top-k K] [--top-p P] [--presence X] [--seed S]\n"
            "                      [--tree N]   (N-node tree; --gamma = max depth)\n"
            "                      [--keyed 0|1] (position-keyed sampling, default 1)\n"
            "                      [--adj 1 [--adj-file F [--adj-ro 1]]] (adjacency drafts)\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        usage();
        return 2;
    }
    if (!strcmp(argv[1], "meta"))
        return cmd_meta(argv[2]);
    if (!strcmp(argv[1], "samptest"))
        return cmd_samptest();
    Q35Model m;
    char err[256];
    const double t0 = now_s();
    if (q35_load(&m, argv[2], err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    const double tl = now_s() - t0;
    int rc = 0;
    if (!strcmp(argv[1], "info")) {
        q35_print_config(&m, stdout);
        printf("  load: %.1f s\n", tl);
    } else if (!strcmp(argv[1], "test") && argc >= 4) {
        rc = cmd_test(&m, argv[3]);
    } else if (!strcmp(argv[1], "treetest")) {
        Opts o;
        if (parse_opts(argc, argv, 3, &o)) {
            q35_free(&m);
            return 2;
        }
        rc = cmd_treetest(&m, &o);
    } else if (!strcmp(argv[1], "run") || !strcmp(argv[1], "score")
               || !strcmp(argv[1], "spec") || (!strcmp(argv[1], "dump") && argc >= 4)) {
        g_model_path = argv[2];
        load_stops(argv[2]);
        const int dump = !strcmp(argv[1], "dump");
        Opts o;
        if (parse_opts(argc, argv, dump ? 4 : 3, &o)) {
            q35_free(&m);
            return 2;
        }
        if (o.print)
            load_vocab(argv[2]);
        Q35State s;
        const int score = !strcmp(argv[1], "score");
        const int spec = !strcmp(argv[1], "spec");
        if (spec && o.chunk < o.gamma + 1)
            o.chunk = o.gamma + 1;
        if (spec && o.chunk < o.tree)
            o.chunk = o.tree;
        if (q35_state_init(&s, &m, o.ctx, o.chunk, o.kv_f16)) {
            fprintf(stderr, "out of memory for state\n");
            q35_free(&m);
            return 1;
        }
        s.state_fmt = o.state;
        s.ro_rmax = o.ro_rank;
        if (o.trace)
            s.trace = fopen(o.trace, "w");
        if (getenv("Q35_DUMP"))
            s.dbg = fopen(getenv("Q35_DUMP"), "wb");
        if (getenv("Q35_HDUMP"))
            s.hdump = fopen(getenv("Q35_HDUMP"), "wb");
        rc = dump ? cmd_dump(&m, &s, &o, argv[3])
             : spec ? cmd_spec(&m, &s, &o) : cmd_run(&m, &s, &o, score);
        if (s.trace)
            fclose(s.trace);
        if (s.dbg)
            fclose(s.dbg);
        if (s.hdump)
            fclose(s.hdump);
        q35_state_free(&s);
    } else {
        usage();
        rc = 2;
    }
    q35_free(&m);
    return rc;
}
