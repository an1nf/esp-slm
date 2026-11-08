/* Draft sources and best-first tree construction. */
#include "cli.h"

const char *ts_name[TR_SRC] = {
    "context", "context (other)", "reservoir", "Jacobi", "Jacobi 2nd", "context 1-gram",
    "seed", "oracle", "adj 1st", "adj 2nd", "adj 3rd-4th",
    "adj 5th-8th", "datastore", "datastore 2nd+"
};
const double ts_prior[TR_SRC] = { 0.6, 0.4, 0.4, 0.3, 0.15, 0.2, 0.05, 0.9,
                                  0.35, 0.15, 0.08, 0.04, 0.5, 0.2 };

int adj_src(int rank)
{
    return rank == 0 ? TS_ADJ0 : rank == 1 ? TS_ADJ1 : rank < 4 ? TS_ADJ2 : TS_ADJ4;
}

int pld(const int *seq, int n, int nmax, int nmin, int gamma, int *out)
{
    for (int k = nmax; k >= nmin; k--) {
        if (n < k + 1)
            continue;
        const int *suf = seq + n - k;
        for (int p = n - k - 1; p >= 0; p--) {
            if (memcmp(seq + p, suf, (size_t)k * sizeof(int)))
                continue;
            int g = 0;
            while (g < gamma && p + k + g < n) {
                out[g] = seq[p + k + g];
                g++;
            }
            if (g > 0)
                return g;
        }
    }
    return 0;
}

PoolEnt g_pool[POOL_N];
int g_pool_n, g_pool_head;

void pool_add(int key, const int *cont, int len)
{
    if (len < 1)
        return;
    if (len > POOL_G)
        len = POOL_G;
    PoolEnt *e = &g_pool[g_pool_head];
    e->key = key;
    e->len = len;
    memcpy(e->cont, cont, (size_t)len * sizeof(int));
    g_pool_head = (g_pool_head + 1) % POOL_N;
    if (g_pool_n < POOL_N)
        g_pool_n++;
}

int pool_get(int key, int gamma, int *out)
{
    for (int i = 1; i <= g_pool_n; i++) {
        const PoolEnt *e = &g_pool[(g_pool_head - i + POOL_N) % POOL_N];
        if (e->key != key)
            continue;
        const int g = e->len < gamma ? e->len : gamma;
        memcpy(out, e->cont, (size_t)g * sizeof(int));
        return g;
    }
    return 0;
}

const float *prefill(const Q35Model *m, Q35State *s, const int *ids, int n, int chunk)
{
    const float *lg = NULL;
    for (int done = 0; done < n;) {
        const int T = n - done < chunk ? n - done : chunk;
        lg = q35_forward(m, s, ids + done, T, 0);
        if (!lg)
            return NULL;
        done += T;
    }
    return lg;
}


/* Online acceptance rate per source and depth, with a prior worth two proposals. */
static double tr_rate(const TreeDraft *R, int src, int d)
{
    const int k = d < 1 ? 0 : d > TR_D ? TR_D - 1 : d - 1;
    return (R->acc[src][k] + 2.0 * ts_prior[src]) / (R->prop[src][k] + 2.0);
}

static int tr_insert(TreeDraft *R, const int *c, int len, int src)
{
    int u = 0, added = 0;
    if (len > R->maxd)
        len = R->maxd;
    for (int i = 0; i < len; i++) {
        int v = R->first[u];
        while (v >= 0 && R->tok[v] != c[i])
            v = R->next[v];
        if (v < 0) {
            if (R->nn >= TR_CAP)
                return added;
            v = R->nn++;
            R->tok[v] = c[i];
            R->par[v] = u;
            R->dep[v] = R->dep[u] + 1;
            R->first[v] = -1;
            R->mask[v] = 0;
            R->expd[v] = 0;
            R->next[v] = R->first[u];
            R->first[u] = v;
            added++;
        }
        R->mask[v] |= 1u << src;
        u = v;
    }
    return added;
}

static int tr_child(TreeDraft *R, int u, int t, int src)
{
    int v = R->first[u];
    while (v >= 0 && R->tok[v] != t)
        v = R->next[v];
    if (v < 0) {
        if (R->nn >= TR_CAP)
            return -1;
        v = R->nn++;
        R->tok[v] = t;
        R->par[v] = u;
        R->dep[v] = R->dep[u] + 1;
        R->first[v] = -1;
        R->mask[v] = 0;
        R->expd[v] = 0;
        R->next[v] = R->first[u];
        R->first[u] = v;
    }
    R->mask[v] |= 1u << src;
    return v;
}

static double tr_q(const TreeDraft *R, int v)
{
    double miss = 1.0;
    for (int sidx = 0; sidx < TR_SRC; sidx++)
        if (R->mask[v] >> sidx & 1u)
            miss *= 1.0 - tr_rate(R, sidx, R->dep[v]);
    return 1.0 - miss;
}

static void tr_emit(TreeDraft *R, int u, int sp)
{
    const int me = R->ns++;
    R->stok[me] = R->tok[u];
    R->spar[me] = sp;
    R->snode[me] = u;

    int ch[256], nc = 0;
    for (int v = R->first[u]; v >= 0 && nc < 256; v = R->next[v])
        if (R->sel[v])
            ch[nc++] = v;
    for (int a = 1; a < nc; a++)
        for (int b = a; b > 0 && R->P[ch[b]] > R->P[ch[b - 1]]; b--) {
            const int t = ch[b];
            ch[b] = ch[b - 1];
            ch[b - 1] = t;
        }
    for (int k = 0; k < nc; k++)
        tr_emit(R, ch[k], me);
}

static const TreeDraft *g_tr_sort;
static int tr_cmp(const void *a, const void *b)
{
    const int x = *(const int *)a, y = *(const int *)b;
    const double px = g_tr_sort->P[x], py = g_tr_sort->P[y];
    if (px != py)
        return px > py ? -1 : 1;
    return x - y;
}

/* Merge all sources into a trie, score P(path), keep the N best nodes whose parents are kept. */
void tr_build(TreeDraft *R, const int *seq, int n, int ngram, const int *orc, int norc)
{
    R->nn = 1;
    R->tok[0] = seq[n - 1];
    R->par[0] = -1;
    R->dep[0] = 0;
    R->first[0] = -1;
    R->next[0] = -1;
    R->mask[0] = 0;
    R->expd[0] = 0;
    int c[TR_MAXD];
    if (orc && norc > 0)
        tr_insert(R, orc, norc, TS_ORC);

    int nlong = 0, nshort = 0;
    for (int k = ngram; k >= 1; k--) {
        if (n < k + 1)
            continue;
        const int *suf = seq + n - k;
        for (int p = n - k - 1; p >= 0; p--) {
            if (k >= 2 ? nlong >= 8 : nshort >= 4)
                break;
            if (memcmp(seq + p, suf, (size_t)k * sizeof(int)))
                continue;
            int g = 0;
            while (g < R->maxd && p + k + g < n) {
                c[g] = seq[p + k + g];
                g++;
            }
            if (g < 1)
                continue;
            const int src = k >= 2 ? (nlong == 0 ? TS_CTX : TS_CTX2) : TS_CTX1;
            if (tr_insert(R, c, g, src) > 0) {
                if (k >= 2)
                    nlong++;
                else
                    nshort++;
            }
        }
    }

    if (R->ds) {
        int dc[4][TR_MAXD], dl[4];
        const int nd = ds_drafts(R->ds, seq, n, R->maxd, dc, dl, 4);
        for (int i = 0; i < nd; i++)
            tr_insert(R, dc[i], dl[i], i ? TS_DS2 : TS_DS);
    }

    int npool = 0;
    for (int i = 1; i <= g_pool_n && npool < 4; i++) {
        const PoolEnt *e = &g_pool[(g_pool_head - i + POOL_N) % POOL_N];
        if (e->key == seq[n - 1] && tr_insert(R, e->cont, e->len, TS_POOL) > 0)
            npool++;
    }

    for (int i = 0; i < R->njc; i++) {
        tr_insert(R, R->jc[i], R->jl[i], TS_JAC);
        if (R->jalt[i] >= 0)
            tr_insert(R, &R->jalt[i], 1, TS_JALT);
    }

    for (int g = 0; g < R->maxd; g++)
        c[g] = seq[n - 1];
    tr_insert(R, c, R->maxd, TS_SEED);

    R->P[0] = 1.0;
    for (int v = 1; v < R->nn; v++)
        R->P[v] = R->P[R->par[v]] * tr_q(R, v);

    if (R->adj) {
        for (int it = 0; it < R->N; it++) {
            int u = -1;
            for (int v = 0; v < R->nn; v++)
                if (!R->expd[v] && R->dep[v] < R->maxd && (u < 0 || R->P[v] > R->P[u]))
                    u = v;
            if (u < 0 || R->P[u] <= 0.0)
                break;
            R->expd[u] = 1;
            const int *row = R->adj + (size_t)R->tok[u] * ADJ_K;
            for (int j = 0; j < ADJ_K && row[j] >= 0; j++) {
                const int v = tr_child(R, u, row[j], adj_src(j));
                if (v < 0)
                    break;
                R->P[v] = R->P[u] * tr_q(R, v);
            }
        }
        for (int v = 1; v < R->nn; v++)
            R->P[v] = R->P[R->par[v]] * tr_q(R, v);
    }

    for (int v = 0; v < R->nn; v++) {
        R->order[v] = v;
        R->sel[v] = 0;
    }
    g_tr_sort = R;
    qsort(R->order + 1, (size_t)(R->nn - 1), sizeof(int), tr_cmp);
    R->sel[0] = 1;
    int picked = 1;
    for (int i = 1; i < R->nn && picked < R->N; i++) {
        const int v = R->order[i];
        if (R->sel[R->par[v]]) {
            R->sel[v] = 1;
            picked++;
        }
    }
    R->ns = 0;
    tr_emit(R, 0, -1);
}

int tr_chain(const TreeDraft *R, const int *y, int u, int *out, int maxlen)
{
    int len = 0;
    while (u >= 0 && len < maxlen) {
        out[len++] = y[u];
        int nx = -1, first = -1;
        for (int v = u + 1; v < R->ns; v++)
            if (R->spar[v] == u) {
                if (first < 0)
                    first = v;
                if (R->stok[v] == y[u]) {
                    nx = v;
                    break;
                }
            }
        u = nx >= 0 ? nx : first;
    }
    return len;
}
