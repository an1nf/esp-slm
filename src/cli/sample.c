/* Sampling, position-keyed noise and multi-draft acceptance. */
#include "cli.h"


double rnd01(Sampler *S)
{
    S->rng ^= S->rng << 13;
    S->rng ^= S->rng >> 7;
    S->rng ^= S->rng << 17;
    return (double)(S->rng >> 11) * (1.0 / 9007199254740992.0);
}

static float plogit(const Sampler *S, const float *lg, int i)
{
    return S->seen[i] ? lg[i] - S->presence : lg[i];
}

static int pick_greedy(const Sampler *S, const float *lg, int V)
{
    int b = 0;
    float bv = plogit(S, lg, 0);
    for (int i = 1; i < V; i++) {
        const float x = plogit(S, lg, i);
        if (x > bv) {
            bv = x;
            b = i;
        }
    }
    return b;
}

void dist_build(Sampler *S, const float *lg, int V)
{
    const int K = S->topk > 0 && S->topk < 256 ? S->topk : 256;
    S->cn = 0;
    float lv[256];
    for (int i = 0; i < V; i++) {
        const float x = plogit(S, lg, i);
        if (S->cn == K && x <= lv[K - 1])
            continue;
        int k = S->cn < K ? S->cn++ : K - 1;
        while (k > 0 && lv[k - 1] < x) {
            lv[k] = lv[k - 1];
            S->cid[k] = S->cid[k - 1];
            k--;
        }
        lv[k] = x;
        S->cid[k] = i;
    }
    double tot = 0.0;
    for (int k = 0; k < S->cn; k++) {
        S->cp[k] = exp(((double)lv[k] - (double)lv[0]) / (double)S->temp);
        tot += S->cp[k];
    }
    double acc = 0.0;
    int keep = S->cn;
    for (int k = 0; k < S->cn; k++) {
        S->cp[k] /= tot;
        acc += S->cp[k];
        if (acc >= (double)S->topp) {
            keep = k + 1;
            break;
        }
    }
    S->cn = keep;
    tot = 0.0;
    for (int k = 0; k < keep; k++)
        tot += S->cp[k];
    for (int k = 0; k < keep; k++)
        S->cp[k] /= tot;
}

double dist_prob(const Sampler *S, int tok)
{
    for (int k = 0; k < S->cn; k++)
        if (S->cid[k] == tok)
            return S->cp[k];
    return 0.0;
}

int dist_sample(Sampler *S, int exclude)
{
    double tot = 0.0;
    for (int k = 0; k < S->cn; k++)
        if (S->cid[k] != exclude)
            tot += S->cp[k];
    if (tot <= 0.0)
        return S->cid[0];
    double u = rnd01(S) * tot;
    for (int k = 0; k < S->cn; k++) {
        if (S->cid[k] == exclude)
            continue;
        u -= S->cp[k];
        if (u <= 0.0)
            return S->cid[k];
    }
    for (int k = S->cn - 1; k >= 0; k--)
        if (S->cid[k] != exclude)
            return S->cid[k];
    return S->cid[0];
}

static unsigned long long mix64(unsigned long long z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double gumbel_key(unsigned long long seed, int pos, int tok)
{
    const unsigned long long h = mix64(mix64(seed ^ 0x5851F42D4C957F2Dull)
                                       ^ mix64((unsigned long long)(unsigned)pos << 1)
                                       ^ ((unsigned long long)(unsigned)tok << 32));
    const double u = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return -log(-log(u));
}

/* Gumbel-max with noise keyed by (seed, position): same text for any draft schedule. */
int keyed_pick(Sampler *S, const float *lg, int V, int pos)
{
    dist_build(S, lg, V);
    int best = S->cid[0];
    double bv = -INFINITY;
    for (int k = 0; k < S->cn; k++) {
        if (S->cp[k] <= 0.0)
            continue;
        const double v = log(S->cp[k]) + gumbel_key(S->seed, pos, S->cid[k]);
        if (v > bv) {
            bv = v;
            best = S->cid[k];
        }
    }
    return best;
}

int choose(Sampler *S, const float *lg, int V, int pos)
{
    if (S->temp <= 0.0f)
        return pick_greedy(S, lg, V);
    if (S->keyed)
        return keyed_pick(S, lg, V, pos);
    dist_build(S, lg, V);
    return dist_sample(S, -1);
}

int second_best(const float *lg, int V, int best)
{
    int b = -1;
    float bv = -INFINITY;
    for (int i = 0; i < V; i++)
        if (i != best && lg[i] > bv) {
            bv = lg[i];
            b = i;
        }
    return b;
}

/* Multi-draft speculative sampling: accept one of nc candidates or draw from the residual. */
int multi_accept(Sampler *S, const float *lg, int V, const int *cand, int nc, int *bonus)
{
    dist_build(S, lg, V);
    double rem = 1.0;
    for (int i = 0; i < nc; i++) {
        const double pv = dist_prob(S, cand[i]);
        if (pv > 0.0 && rnd01(S) * rem < pv)
            return i;
        for (int k = 0; k < S->cn; k++)
            if (S->cid[k] == cand[i]) {
                rem -= S->cp[k];
                S->cp[k] = 0.0;
            }
    }
    *bonus = dist_sample(S, -1);
    return -1;
}

int cmd_samptest(void)
{
    const int V = 40, trials = 400000;
    float lg[40];
    unsigned long long r = 99;
    for (int i = 0; i < V; i++) {
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        lg[i] = (float)((r >> 11) % 1000) / 250.0f;
    }
    unsigned char seen[40] = {0};
    const struct { float temp, topp; int topk, nc, cand[5]; } cases[] = {
        { 1.0f, 1.0f, 0, 3, { 7, 3, 21 } },
        { 0.7f, 0.9f, 8, 4, { 0, 5, 9, 12 } },
        { 1.5f, 1.0f, 5, 5, { 30, 1, 2, 3, 4 } },
    };
    int fails = 0;
    for (int c = 0; c < 3; c++) {
        Sampler S;
        memset(&S, 0, sizeof(S));
        S.temp = cases[c].temp;
        S.topp = cases[c].topp;
        S.topk = cases[c].topk;
        S.seen = seen;
        S.rng = 0x9E3779B97F4A7C15ull + (unsigned long long)c;

        dist_build(&S, lg, V);
        double p[40] = {0};
        for (int k = 0; k < S.cn; k++)
            p[S.cid[k]] = S.cp[k];
        int cand[5];
        for (int i = 0; i < cases[c].nc; i++)
            cand[i] = cases[c].cand[i] < S.cn ? S.cid[cases[c].cand[i]] : (S.cid[0] + 1 + cases[c].cand[i]) % V;
        long cnt[40] = {0};
        for (int t = 0; t < trials; t++) {
            int b;
            const int k = multi_accept(&S, lg, V, cand, cases[c].nc, &b);
            cnt[k >= 0 ? cand[k] : b]++;
        }
        double chi = 0.0;
        int dof = -1, bad = 0;
        for (int i = 0; i < V; i++) {
            const double e = p[i] * trials;
            if (e == 0.0) {
                bad += cnt[i] != 0;
                continue;
            }
            chi += (cnt[i] - e) * (cnt[i] - e) / e;
            dof++;
        }

        const double lim = dof + 5.0 * sqrt(2.0 * dof);
        const int ok = !bad && chi < lim;
        printf("  case %d (temp %.1f, top-k %d, top-p %.1f, %d drafts): chi2 %.1f, df %d, "
               "limit %.1f, off-support %d: %s\n", c, cases[c].temp, cases[c].topk,
               cases[c].topp, cases[c].nc, chi, dof, lim, bad, ok ? "OK" : "FAIL");
        fails += !ok;
    }

    for (int c = 0; c < 2; c++) {
        Sampler S;
        memset(&S, 0, sizeof(S));
        S.temp = c ? 0.7f : 1.2f;
        S.topp = c ? 0.9f : 1.0f;
        S.topk = c ? 8 : 0;
        S.seen = seen;
        dist_build(&S, lg, V);
        double p[40] = {0};
        for (int k = 0; k < S.cn; k++)
            p[S.cid[k]] = S.cp[k];
        long cnt[40] = {0};
        const int tr = trials / 2;
        for (int t = 0; t < tr; t++) {
            S.seed = 1000003ull * (unsigned long long)t + 17ull * (unsigned long long)c;
            cnt[keyed_pick(&S, lg, V, 12345 + c)]++;
        }
        double chi = 0.0;
        int dof = -1, bad = 0;
        for (int i = 0; i < V; i++) {
            const double e = p[i] * tr;
            if (e == 0.0) {
                bad += cnt[i] != 0;
                continue;
            }
            chi += (cnt[i] - e) * (cnt[i] - e) / e;
            dof++;
        }
        const double lim = dof + 5.0 * sqrt(2.0 * dof);
        const int ok = !bad && chi < lim;
        printf("  keyed %d (temp %.1f, top-k %d, top-p %.1f): chi2 %.1f, df %d, "
               "limit %.1f, off-support %d: %s\n", c, S.temp, S.topk, S.topp, chi, dof,
               lim, bad, ok ? "OK" : "FAIL");
        fails += !ok;
    }
    printf("multi-draft sampling: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
