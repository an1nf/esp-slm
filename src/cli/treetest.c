/* Tree evaluation vs sequential decoding. */
#include "cli.h"

int cmd_treetest(const Q35Model *m, const Opts *o)
{
    int *ids = NULL;
    const int n0 = parse_ids(o->ids, &ids);
    if (n0 < 1) {
        fprintf(stderr, "treetest: --ids required\n");
        return 2;
    }
    const int N = o->gamma > 1 ? o->gamma : 12, V = m->c.vocab;
    const int maxT = N > o->chunk ? N : o->chunk, rounds = 6;
    unsigned long long r = o->seed * 2654435761ULL + 12345;
#define TT_RND() (r ^= r << 13, r ^= r >> 7, r ^= r << 17, r)
    Q35State A, B;
    if (q35_state_init(&A, m, o->ctx, maxT, o->kv_f16)
        || q35_state_init(&B, m, o->ctx, maxT, o->kv_f16)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    A.state_fmt = B.state_fmt = o->state;
    A.ro_rmax = B.ro_rmax = o->ro_rank;

    const int ro = o->state >= Q35_STATE_RO32;
    const float tol = ro ? (o->tol > 0.0f ? o->tol : 0.05f) : 0.0f;
    if (ro)
        B.state_fmt = Q35_STATE_F32;
    double maxdiff = 0.0;
    long amax_ok = 0, amax_n = 0;
    int *hist = (int *)malloc(((size_t)n0 + (size_t)rounds * N + 1) * sizeof(int));
    int *tok = (int *)malloc((size_t)N * sizeof(int)), *par = (int *)malloc((size_t)N * sizeof(int));
    int *path = (int *)malloc((size_t)N * sizeof(int)), *leaf = (int *)calloc((size_t)N, sizeof(int));
    float *LA = (float *)malloc((size_t)N * V * sizeof(float));
    memcpy(hist, ids, (size_t)n0 * sizeof(int));
    int nh = n0, fails = 0, checked = 0;
    if (!prefill(m, &A, ids, n0, o->chunk))
        return 1;
    for (int rd = 0; rd < rounds; rd++) {
        const int chain = rd % 3 == 2;
        int maxd = 0;
        for (int i = 0; i < N; i++) {
            tok[i] = (int)(TT_RND() % (unsigned long long)V);
            if (chain || i == 0) {
                par[i] = i - 1;
            } else {
                int u = i - 1, steps = (int)(TT_RND() % 3);
                while (steps-- > 0 && u >= 0)
                    u = par[u];
                par[i] = u;
            }
        }
        const float *lg = chain ? q35_eval(m, &A, tok, N, 1) : q35_eval_tree(m, &A, tok, par, N);
        if (!lg) {
            fprintf(stderr, "treetest: evaluation failed\n");
            return 1;
        }
        memcpy(LA, lg, (size_t)N * V * sizeof(float));
        for (int i = 0; i < N; i++)
            leaf[i] = 1;
        for (int i = 0; i < N; i++)
            if (par[i] >= 0)
                leaf[par[i]] = 0;
        int bad = 0;
        for (int i = 0; i < N; i++) {
            if (!leaf[i])
                continue;
            int d = 0;
            for (int u = i; u >= 0; u = par[u])
                path[d++] = u;
            if (d > maxd)
                maxd = d;
            q35_reset(&B, m);
            const float *lb = prefill(m, &B, hist, nh, o->chunk);
            for (int k = d - 1; k >= 0 && lb; k--) {
                const int u = path[k];
                lb = q35_forward(m, &B, &tok[u], 1, 0);
                checked++;
                if (lb && ro) {
                    double md = 0;
                    for (int j = 0; j < V; j++) {
                        const double e = fabs((double)lb[j] - LA[(size_t)u * V + j]);
                        if (e > md)
                            md = e;
                    }
                    if (md > maxdiff)
                        maxdiff = md;
                    amax_n++;
                    amax_ok += argmax(lb, V) == argmax(LA + (size_t)u * V, V);
                    if (md > tol) {
                        if (!bad)
                            printf("  round %d node %d: max diff %.3g > %.3g\n", rd, u, md, (double)tol);
                        bad++;
                    }
                } else if (lb && memcmp(lb, LA + (size_t)u * V, (size_t)V * sizeof(float))) {
                    double md = 0;
                    for (int j = 0; j < V; j++) {
                        const double e = fabs((double)lb[j] - LA[(size_t)u * V + j]);
                        if (e > md)
                            md = e;
                    }
                    if (!bad)
                        printf("  round %d node %d (depth %d): differs, max %.3g\n", rd, u, d - 1 - k, md);
                    bad++;
                }
            }
        }

        const int c = (int)(TT_RND() % (unsigned long long)N);
        int d = 0;
        for (int u = c; u >= 0; u = par[u])
            path[d++] = u;
        for (int k = 0; k < d / 2; k++) {
            const int t = path[k];
            path[k] = path[d - 1 - k];
            path[d - 1 - k] = t;
        }
        if (q35_commit_path(m, &A, path, d)) {
            fprintf(stderr, "treetest: commit_path rejected a valid path\n");
            return 1;
        }
        for (int k = 0; k < d; k++)
            hist[nh++] = tok[path[k]];
        printf("  round %d: %s of %d nodes, max depth %d, %s, commit %d\n", rd,
               chain ? "chain" : "tree", N, maxd, bad ? "FAIL" : "ok", d);
        fails += bad;
    }

    q35_reset(&B, m);
    const float *lb = prefill(m, &B, hist, nh, o->chunk);
    const int nx = (int)(TT_RND() % (unsigned long long)V);
    lb = q35_forward(m, &B, &nx, 1, 0);
    const float *la = q35_forward(m, &A, &nx, 1, 0);
    checked++;
    double mdl = 0.0;
    if (la && lb && ro)
        for (int j = 0; j < V; j++)
            if (fabs((double)la[j] - lb[j]) > mdl)
                mdl = fabs((double)la[j] - lb[j]);
    if (!la || !lb || (ro ? mdl > tol : memcmp(la, lb, (size_t)V * sizeof(float)) != 0)) {
        printf("  state after commit: differs\n");
        fails++;
    }

    int rej = 0;
    {
        const float *lg = q35_eval_tree(m, &A, tok, par, N);
        int bp[2] = { 1, 0 };
        rej += lg && q35_commit_path(m, &A, bp, 2) != 0;
        int bad_par[3] = { -1, 0, 1 };
        bad_par[2] = 2;
        rej += q35_eval_tree(m, &A, tok, bad_par, 3) == NULL;
        int np[4] = { -1, -1, 0, -1 };
        rej += N >= 4 ? q35_eval_tree(m, &A, tok, np, 4) == NULL : 1;
    }
    if (rej != 3) {
        printf("  invalid inputs accepted\n");
        fails++;
    }
    if (ro)
        printf("%d checks against exact f32 (%llu materializations): max diff %.3g "
               "(tol %.3g), argmax %ld/%ld; %s\n", checked, (unsigned long long)A.ro_mat,
               maxdiff, (double)tol, amax_ok, amax_n, fails ? "FAIL" : "OK");
    else
        printf("%d logit checks; %s\n", checked,
               fails ? "FAIL" : "OK: tree bit-exact with sequential decoding");
#undef TT_RND
    q35_state_free(&A);
    q35_state_free(&B);
    free(hist); free(tok); free(par); free(path); free(leaf); free(LA); free(ids);
    return fails ? 1 : 0;
}
