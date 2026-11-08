/* Speculative decoding: chain and tree verification with weight-free drafts. */
#include "cli.h"

/* One verification pass over the draft tree; returns tokens accepted. */
static int tree_pass(const Q35Model *m, Q35State *s, TreeDraft *R, Sampler *SM,
                     const int *seq, int n, int ngram, const int *orc, int norc,
                     int *acc_out, int *nacc, int *bonus_out, int *y)
{
    const int V = m->c.vocab;
    tr_build(R, seq, n, ngram, orc, norc);
    const float *lg = q35_eval_tree(m, s, R->stok, R->spar, R->ns);
    if (!lg)
        return -1;
    const int N = R->ns;
    for (int i = 0; i < N; i++)
        y[i] = argmax(lg + (size_t)i * V, V);
    if (R->adj)
        for (int i = 0; i < N; i++) {
            const float *li = lg + (size_t)i * V;
            int ti[ADJ_K];
            float tv[ADJ_K];
            int nt = 0;
            for (int t = 0; t < V; t++) {
                const float x = li[t];
                if (nt == ADJ_K && x <= tv[ADJ_K - 1])
                    continue;
                int k = nt < ADJ_K ? nt++ : ADJ_K - 1;
                while (k > 0 && tv[k - 1] < x) {
                    tv[k] = tv[k - 1];
                    ti[k] = ti[k - 1];
                    k--;
                }
                tv[k] = x;
                ti[k] = t;
            }
            memcpy(R->adj + (size_t)R->stok[i] * ADJ_K, ti, sizeof(ti));
        }
    int path[TR_MAXD + 1], np = 0, u = 0, bonus = -1;
    path[np++] = 0;
    for (;;) {
        const float *lu = lg + (size_t)u * V;
        int nxt = -1;
        if (SM->temp <= 0.0f || SM->keyed) {
            const int t = choose(SM, lu, V, n + R->dep[R->snode[u]]);
            for (int v = u + 1; v < N; v++)
                if (R->spar[v] == u && R->stok[v] == t) {
                    nxt = v;
                    break;
                }
            bonus = t;
        } else {
            int cand[256], cnode[256], nc = 0;
            for (int v = u + 1; v < N && nc < 256; v++)
                if (R->spar[v] == u) {
                    cnode[nc] = v;
                    cand[nc++] = R->stok[v];
                }
            const int k = multi_accept(SM, lu, V, cand, nc, &bonus);
            nxt = k >= 0 ? cnode[k] : -1;
        }
        if (nxt < 0)
            break;
        SM->seen[R->stok[nxt]] = 1;
        path[np++] = nxt;
        u = nxt;
    }

    unsigned char onp[TR_CAP];
    memset(onp, 0, (size_t)N);
    for (int k = 0; k < np; k++)
        onp[path[k]] = 1;
    for (int v = 1; v < N; v++) {
        if (!onp[R->spar[v]])
            continue;
        const int tn = R->snode[v], d = R->dep[tn];
        const int k = d > TR_D ? TR_D - 1 : d - 1;
        for (int sidx = 0; sidx < TR_SRC; sidx++)
            if (R->mask[tn] >> sidx & 1u) {
                R->prop[sidx][k] += 1.0;
                R->tot_prop[sidx]++;
                if (onp[v]) {
                    R->acc[sidx][k] += 1.0;
                    R->tot_acc[sidx]++;
                }
            }
    }
    R->hist_acc[np - 1 < TR_MAXD ? np - 1 : TR_MAXD]++;

    int *yj = y;
    int yk[TR_CAP];
    if (SM->temp > 0.0f && SM->keyed) {
        SM->seen[bonus] = 1;
        memcpy(yk, y, (size_t)N * sizeof(int));
        for (int x = u + 1; x < N; x++) {
            int a = R->spar[x];
            while (a >= 0 && a != u)
                a = R->spar[a];
            if (a == u)
                yk[x] = keyed_pick(SM, lg + (size_t)x * V, V, n + R->dep[R->snode[x]]);
        }
        yj = yk;
    }
    R->njc = 0;
    for (int v = u + 1; v < N && R->njc < TR_JC; v++)
        if (R->spar[v] == u) {
            const int i = R->njc++;
            R->jl[i] = tr_chain(R, yj, v, R->jc[i], R->maxd);
            R->jalt[i] = second_best(lg + (size_t)v * V, V, yj[v]);
        }

    int cbuf[POOL_G];
    for (int v = 1; v < N; v++)
        if (!onp[v])
            pool_add(R->stok[v], cbuf, tr_chain(R, y, v, cbuf, POOL_G));
    if (q35_commit_path(m, s, path, np))
        return -1;
    *nacc = np - 1;
    for (int k = 1; k < np; k++)
        acc_out[k - 1] = R->stok[path[k]];
    *bonus_out = bonus;
    return N;
}

int cmd_spec(const Q35Model *m, Q35State *s, const Opts *o)
{
    int *ids;
    const int n0 = o->ids ? parse_ids(o->ids, &ids) : -1;
    const int V = m->c.vocab;
    const int lanes = o->tree > o->gamma + 1 ? o->tree : o->gamma + 1;
    if (n0 < 1 || o->gen < 1 || n0 + o->gen + lanes > s->max_seq
        || o->gamma + 1 > s->max_T) {
        fprintf(stderr, "need --ids and --gen, --ctx >= prompt + gen + gamma + 1 "
                        "and --chunk >= gamma + 1\n");
        return 1;
    }
    int *seq = (int *)malloc((size_t)(n0 + o->gen + o->gamma + 2) * sizeof(int));
    int *d = (int *)malloc((size_t)(o->gamma + 1) * sizeof(int));
    int *toks = (int *)malloc((size_t)(o->gamma + 2) * sizeof(int));
    if (!seq || !d || !toks)
        return 1;
    memcpy(seq, ids, (size_t)n0 * sizeof(int));
    if (o->print || o->stream)
        load_vocab_once(o);
    const double t0 = now_s();
    const float *lg = prefill(m, s, ids, n0, s->max_T);
    if (!lg)
        return 1;
    const double t1 = now_s();
    Sampler SM;
    memset(&SM, 0, sizeof(SM));
    SM.temp = o->temp;
    SM.topp = o->topp > 0.0f && o->topp <= 1.0f ? o->topp : 1.0f;
    SM.topk = o->topk;
    SM.presence = o->presence;
    SM.rng = o->seed ? o->seed * 0x9E3779B97F4A7C15ull : 0x9E3779B97F4A7C15ull;
    SM.keyed = o->keyed;
    SM.seed = o->seed;
    SM.seen = (unsigned char *)calloc((size_t)V, 1);
    if (!SM.seen)
        return 1;
    int n = n0;
    seq[n] = choose(&SM, lg, V, n);
    SM.seen[seq[n]] = 1;
    n++;
    int shown = n0;
#define STREAM_UPTO(lim)                                                         \
    do {                                                                         \
        if (o->stream) {                                                         \
            for (; shown < (lim); shown++)                                       \
                if (!is_stop(seq[shown]))                                        \
                    print_token(stderr, seq[shown]);                             \
            fflush(stderr);                                                      \
        }                                                                        \
    } while (0)
    STREAM_UPTO(n);
    int produced = 1, passes = 0, drafted = 0, accepted = 0, tsum = 0;
    const int stop_now = o->stop && is_stop(seq[n - 1]);
    int *orc = NULL, norc = 0;
    if (o->oracle)
        norc = parse_ids(o->oracle, &orc);
    uint32_t rng = 12345u;
    const int ylen = (o->gamma + 1 > o->tree ? o->gamma + 1 : o->tree);
    int *y = (int *)malloc((size_t)ylen * sizeof(int));
    TreeDraft *TR = NULL;
    if (o->tree > 0) {
        if (o->tree > s->max_T || o->tree > 256 || o->gamma > TR_MAXD) {
            fprintf(stderr, "--tree <= min(chunk, 256) and --gamma <= %d\n", TR_MAXD);
            return 1;
        }
        TR = (TreeDraft *)calloc(1, sizeof(TreeDraft));
        if (!TR)
            return 1;
        TR->N = o->tree;
        TR->maxd = o->gamma;
        if (o->ds) {
            Datastore *D = (Datastore *)calloc(1, sizeof(Datastore));
            if (!D || ds_load(D, o->ds, V)) {
                fprintf(stderr, "cannot load datastore %s\n", o->ds);
                return 1;
            }
            fprintf(stderr, "datastore: %d tokens (%s)\n", D->n, o->ds);
            TR->ds = D;
        }
        if (o->adj) {
            TR->V = V;
            TR->adj = (int *)malloc((size_t)V * ADJ_K * sizeof(int));
            if (!TR->adj)
                return 1;
            for (size_t i = 0; i < (size_t)V * ADJ_K; i++)
                TR->adj[i] = -1;
            FILE *f = o->adj_file ? fopen(o->adj_file, "rb") : NULL;
            if (f) {
                int hdr[3] = {0, 0, 0};
                if (fread(hdr, sizeof(int), 3, f) == 3 && hdr[0] == 0x4A444151 && hdr[1] == V
                    && hdr[2] == ADJ_K
                    && fread(TR->adj, sizeof(int), (size_t)V * ADJ_K, f) == (size_t)V * ADJ_K) {
                    long filled = 0;
                    for (int t = 0; t < V; t++)
                        filled += TR->adj[(size_t)t * ADJ_K] >= 0;
                    fprintf(stderr, "adjacency: %ld tokens with candidates (%s)\n", filled, o->adj_file);
                } else {
                    fprintf(stderr, "adjacency: %s does not match this model, starting empty\n",
                            o->adj_file);
                    for (size_t i = 0; i < (size_t)V * ADJ_K; i++)
                        TR->adj[i] = -1;
                }
                fclose(f);
            }
        }
    }
    int *jac = (int *)malloc((size_t)(o->gamma + 1) * sizeof(int));
    int njac = 0, src[4] = {0, 0, 0, 0}, src_acc[4] = {0, 0, 0, 0};
    g_pool_n = g_pool_head = 0;
    while (!stop_now && produced < o->gen) {
        int g = 0, which = 3;
        if (orc) {
            for (int i = n - n0; i < norc && g < o->gamma; i++) {
                rng = rng * 1664525u + 1013904223u;
                const int bad = (double)(rng >> 8) / 16777216.0 < o->corrupt;
                d[g++] = bad ? (orc[i] + 1) % V : orc[i];
            }
        } else if (o->tree > 0) {
        } else if (o->drafter == DR_PLD) {
            g = pld(seq, n, o->ngram, 1, o->gamma, d);
            which = 0;
        } else if (o->drafter == DR_JACOBI) {
            g = njac < o->gamma ? njac : o->gamma;
            memcpy(d, jac, (size_t)g * sizeof(int));
            which = 2;
            if (g == 0) {
                g = pld(seq, n, 1, 1, o->gamma, d);
                for (; g < o->gamma; g++)
                    d[g] = seq[n - 1];
            }
        } else if (o->drafter == DR_MIX) {
            if ((g = pld(seq, n, o->ngram, 2, o->gamma, d)) > 0)
                which = 0;
            else if ((g = pool_get(seq[n - 1], o->gamma, d)) > 0)
                which = 1;
            else if ((g = njac < o->gamma ? njac : o->gamma) > 0) {
                memcpy(d, jac, (size_t)g * sizeof(int));
                which = 2;
            } else {
                g = pld(seq, n, 1, 1, o->gamma, d);
                for (; g < o->gamma; g++)
                    d[g] = seq[n - 1];
                which = 2;
            }
        }
        int j = 0, bonus, T;
        if (TR) {
            T = tree_pass(m, s, TR, &SM, seq, n, o->ngram, orc ? d : NULL, orc ? g : 0,
                          d, &j, &bonus, y);
            if (T < 0)
                return 1;
            g = T - 1;
            for (int i = 0; i < j; i++)
                seq[n++] = d[i];
            seq[n++] = bonus;
            SM.seen[bonus] = 1;
            goto pass_done;
        }
        toks[0] = seq[n - 1];
        memcpy(toks + 1, d, (size_t)g * sizeof(int));
        T = g + 1;
        lg = q35_eval(m, s, toks, T, 1);
        if (!lg)
            return 1;
        for (int i = 0; i < T; i++)
            y[i] = argmax(lg + (size_t)i * V, V);

        for (;;) {
            const float *lj = lg + (size_t)j * V;
            if (SM.temp <= 0.0f || SM.keyed) {
                const int t = choose(&SM, lj, V, n + j);
                if (j < g && t == d[j]) {
                    SM.seen[d[j++]] = 1;
                    continue;
                }
                bonus = t;
                break;
            }
            dist_build(&SM, lj, V);
            if (j < g) {
                if (rnd01(&SM) < dist_prob(&SM, d[j])) {
                    SM.seen[d[j++]] = 1;
                    continue;
                }
                bonus = dist_sample(&SM, d[j]);
            } else {
                bonus = dist_sample(&SM, -1);
            }
            break;
        }
        for (int i = 0; i < j; i++)
            seq[n++] = d[i];
        seq[n++] = bonus;
        SM.seen[bonus] = 1;
        y[j] = bonus;
        q35_commit(s, j + 1);

        njac = T - (j + 1);
        memcpy(jac, y + j + 1, (size_t)(njac > 0 ? njac : 0) * sizeof(int));
        if (njac < 0)
            njac = 0;
        if (SM.temp > 0.0f && SM.keyed)
            for (int i = 0; i < njac; i++)
                jac[i] = keyed_pick(&SM, lg + (size_t)(j + 1 + i) * V, V, n - (j + 1) + j + 1 + i);
        for (int i = j + 1; i < T; i++)
            pool_add(toks[i], y + i, T - i);
        src[which] += g;
        src_acc[which] += j;
    pass_done:
        produced += j + 1;
        passes++;
        drafted += g;
        accepted += j;
        tsum += T;
        if (o->stop) {
            int cut = -1;
            for (int i = n - (j + 1); i < n; i++)
                if (is_stop(seq[i])) {
                    cut = i;
                    break;
                }
            if (cut >= 0) {
                n = cut + 1;
                produced = n - n0;
                STREAM_UPTO(n);
                break;
            }
        }
        STREAM_UPTO(n);
    }
#undef STREAM_UPTO
    if (o->stream)
        fputc('\n', stderr);
    const double t2 = now_s();
    const int gen = produced;
    printf("spec: %d tokens in %d passes (%.2f tokens/pass); drafted %d, accepted %d "
           "(%.2f); mean T %.2f\n", gen - 1, passes, passes ? (double)(gen - 1) / passes : 0.0,
           drafted, accepted, drafted ? (double)accepted / drafted : 0.0,
           passes ? (double)tsum / passes : 0.0);
    printf("time: prefill %.2f s, decode %.2f s (%.3f s/token)\n", t1 - t0, t2 - t1,
           gen > 1 ? (t2 - t1) / (gen - 1) : 0.0);
    if (TR) {
        printf("tree of %d nodes, depth <= %d; accepted/proposed (parent accepted) "
               "by source:\n ", TR->N, TR->maxd);
        for (int k = 0; k < TR_SRC; k++)
            if (TR->tot_prop[k])
                printf(" %s %ld/%ld;", ts_name[k], TR->tot_acc[k], TR->tot_prop[k]);
        printf("\n  accepted drafts per pass:");
        for (int k = 0; k <= TR_MAXD; k++)
            if (TR->hist_acc[k])
                printf(" %d:%ld", k, TR->hist_acc[k]);
        printf("\n");
    } else {
        printf("accepted/proposed by source: context %d/%d, reservoir %d/%d, Jacobi %d/%d\n",
               src_acc[0], src[0], src_acc[1], src[1], src_acc[2], src[2]);
    }
    printf("ids: ");
    for (int i = n0; i < n0 + gen; i++)
        printf("%s%d", i > n0 ? "," : "", seq[i]);
    printf("\n");
    if (o->print) {
        for (int i = n0; i < n0 + gen; i++)
            print_token(stdout, seq[i]);
        printf("\n");
    }
    int rc = 0;
    if (o->check && SM.temp > 0.0f && !SM.keyed) {
        printf("CHECK skipped: unkeyed sampling is not reproducible\n");
    } else if (o->check) {
        q35_reset(s, m);
        memset(SM.seen, 0, (size_t)V);
        const float *l2 = prefill(m, s, ids, n0, 1);
        int bad = -1;
        for (int i = 0; i < gen && l2; i++) {
            const int t = choose(&SM, l2, V, n0 + i);
            SM.seen[t] = 1;
            if (t != seq[n0 + i]) {
                bad = i;
                break;
            }
            if (i + 1 < gen)
                l2 = q35_forward(m, s, &t, 1, 0);
        }
        if (bad >= 0) {
            printf("CHECK FAIL: differs at token %d\n", bad);
            rc = 1;
        } else {
            printf("CHECK OK: matches sequential decoding\n");
        }
    }
    free(SM.seen);
    free(orc);
    if (TR && TR->adj && o->adj_file && !o->adj_ro) {
        FILE *f = fopen(o->adj_file, "wb");
        const int hdr[3] = { 0x4A444151, V, ADJ_K };
        if (!f || fwrite(hdr, sizeof(int), 3, f) != 3
            || fwrite(TR->adj, sizeof(int), (size_t)V * ADJ_K, f) != (size_t)V * ADJ_K)
            fprintf(stderr, "adjacency: cannot save %s\n", o->adj_file);
        if (f)
            fclose(f);
    }
    if (TR) {
        free(TR->adj);
        if (TR->ds) {
            ds_free((Datastore *)TR->ds);
            free((void *)TR->ds);
        }
    }
    free(TR);
    free(y);
    free(jac);
    free(seq);
    free(d);
    free(toks);
    free(ids);
    return rc;
}
