/* Suffix-array datastore drafts. */
#include "cli.h"

/* Suffix-array datastore: HDS1 header, n, vocab, int32 tokens[n], int32 sa[n]. Separator -1. */
int ds_load(Datastore *D, const char *path, int V)
{
    memset(D, 0, sizeof(*D));
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    char mg[4];
    uint32_t h[2];
    int ok = fread(mg, 1, 4, f) == 4 && !memcmp(mg, "HDS1", 4) && fread(h, 4, 2, f) == 2
             && (h[1] == 0 || (int)h[1] == V) && h[0] > 0 && h[0] < (1u << 30);
    if (ok) {
        D->n = (int)h[0];
        D->tok = (int32_t *)malloc((size_t)D->n * 4);
        D->sa = (int32_t *)malloc((size_t)D->n * 4);
        ok = D->tok && D->sa && fread(D->tok, 4, (size_t)D->n, f) == (size_t)D->n
             && fread(D->sa, 4, (size_t)D->n, f) == (size_t)D->n;
        for (int i = 0; ok && i < D->n; i++)
            ok = D->tok[i] >= -1 && D->tok[i] < V && D->sa[i] >= 0 && D->sa[i] < D->n;
    }
    fclose(f);
    if (!ok)
        ds_free(D);
    return ok ? 0 : -1;
}

void ds_free(Datastore *D)
{
    free(D->tok);
    free(D->sa);
    memset(D, 0, sizeof(*D));
}

static int ds_cmp(const Datastore *D, int p, const int *pat, int k)
{
    for (int i = 0; i < k; i++) {
        if (p + i >= D->n)
            return -1;
        const int a = D->tok[p + i], b = pat[i];
        if (a != b)
            return a < b ? -1 : 1;
    }
    return 0;
}

/* SA range [lo, hi) of suffixes starting with pat[0..k). */
static int ds_range(const Datastore *D, const int *pat, int k, int *lo, int *hi)
{
    int a = 0, b = D->n;
    while (a < b) {
        const int mid = a + (b - a) / 2;
        if (ds_cmp(D, D->sa[mid], pat, k) < 0)
            a = mid + 1;
        else
            b = mid;
    }
    *lo = a;
    b = D->n;
    while (a < b) {
        const int mid = a + (b - a) / 2;
        if (ds_cmp(D, D->sa[mid], pat, k) <= 0)
            a = mid + 1;
        else
            b = mid;
    }
    *hi = a;
    return *hi > *lo;
}

/* Continuations of the longest context suffix found, grouped by next token (most frequent first). */
int ds_drafts(const Datastore *D, const int *seq, int n, int maxd, int out[][TR_MAXD], int *len,
              int maxc)
{
    enum { KMAX = 12, SAMPLES = 64 };
    int lo = 0, hi = 0, k;
    for (k = n < KMAX ? n : KMAX; k >= 1; k--)
        if (ds_range(D, seq + n - k, k, &lo, &hi))
            break;
    if (k < 1)
        return 0;
    int first[SAMPLES], cnt[SAMPLES], at[SAMPLES], ng = 0;
    const int span = hi - lo, step = span > SAMPLES ? span / SAMPLES : 1;
    for (int i = lo; i < hi && ng < SAMPLES; i += step) {
        const int p = D->sa[i] + k;
        if (p >= D->n || D->tok[p] < 0)
            continue;
        int g = 0;
        while (g < ng && first[g] != D->tok[p])
            g++;
        if (g == ng) {
            first[ng] = D->tok[p];
            cnt[ng] = 0;
            at[ng++] = p;
        }
        cnt[g]++;
    }
    int nc = 0;
    while (nc < maxc) {
        int b = -1;
        for (int g = 0; g < ng; g++)
            if (cnt[g] > 0 && (b < 0 || cnt[g] > cnt[b]))
                b = g;
        if (b < 0)
            break;
        int l = 0;
        for (int p = at[b]; l < maxd && p < D->n && D->tok[p] >= 0; p++)
            out[nc][l++] = D->tok[p];
        len[nc++] = l;
        cnt[b] = 0;
    }
    return nc;
}
