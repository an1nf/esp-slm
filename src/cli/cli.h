/* q35 CLI: options, samplers and draft trees. */
#ifndef CLI_H
#define CLI_H

#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "q35.h"

typedef struct {
    const char *ids, *trace;
    int gen, chunk, top, ctx, kv_f16, print, from, state, gamma, ngram, check;
    const char *oracle;
    double corrupt;
    int drafter;
    int stop;
    int stream;
    float temp, topp, presence;
    int topk;
    unsigned long long seed;
    int tree;
    int keyed;
    int adj;
    const char *adj_file;
    int adj_ro;
    int ro_rank;
    float tol;
    int dither, smooth, x16, kv_bits, kv_refine, kv_recent;
    float hi;
    const char *ds;
} Opts;


enum { DR_PLD, DR_JACOBI, DR_MIX, DR_NONE };

typedef struct {
    float temp, topp, presence;
    int topk;
    unsigned long long rng;
    unsigned char *seen;
    int cid[256], cn;
    double cp[256];
    int keyed;
    unsigned long long seed;
} Sampler;

#define POOL_N 4096
#define POOL_G 16
typedef struct {
    int key, len;
    int cont[POOL_G];
} PoolEnt;

enum { TS_CTX, TS_CTX2, TS_POOL, TS_JAC, TS_JALT, TS_CTX1, TS_SEED, TS_ORC,
       TS_ADJ0, TS_ADJ1, TS_ADJ2, TS_ADJ4, TS_DS, TS_DS2, TR_SRC };

#define ADJ_K 8

#define TR_D 16
#define TR_CAP 2048
#define TR_MAXD 64
#define TR_JC 4

typedef struct {
    int N, maxd;
    int nn, tok[TR_CAP], par[TR_CAP], first[TR_CAP], next[TR_CAP], dep[TR_CAP], sel[TR_CAP];
    unsigned mask[TR_CAP];
    double P[TR_CAP];
    int order[TR_CAP];

    int ns, stok[TR_CAP], spar[TR_CAP], snode[TR_CAP];

    double prop[TR_SRC][TR_D], acc[TR_SRC][TR_D];
    long tot_prop[TR_SRC], tot_acc[TR_SRC];

    int njc, jc[TR_JC][TR_MAXD], jl[TR_JC], jalt[TR_JC];
    long hist_acc[TR_MAXD + 1];
    int *adj, V;
    int expd[TR_CAP];
    const struct Datastore *ds;
} TreeDraft;

typedef struct Datastore {
    int n;
    int32_t *tok, *sa;
} Datastore;

int ds_load(Datastore *D, const char *path, int V);
void ds_free(Datastore *D);
int ds_drafts(const Datastore *D, const int *seq, int n, int maxd, int out[][TR_MAXD], int *len,
              int maxc);

extern PoolEnt g_pool[POOL_N];
extern int g_pool_n, g_pool_head;
extern const char *ts_name[TR_SRC];
extern const double ts_prior[TR_SRC];
int adj_src(int rank);


extern const char *g_model_path;

const float *prefill(const Q35Model *m, Q35State *s, const int *ids, int n, int chunk);
double dist_prob(const Sampler *S, int tok);
double now_s(void);
double rnd01(Sampler *S);
int argmax(const float *lg, int V);
int choose(Sampler *S, const float *lg, int V, int pos);
int cmd_cmpstate(const Q35Model *m, const Opts *o, const char *cands);
int cmd_dump(const Q35Model *m, Q35State *s, const Opts *o, const char *out);
int cmd_meta(const char *path);
int cmd_run(const Q35Model *m, Q35State *s, const Opts *o, int score);
int cmd_samptest(void);
int cmd_spec(const Q35Model *m, Q35State *s, const Opts *o);
int cmd_test(const Q35Model *m, const char *ref);
int cmd_treetest(const Q35Model *m, const Opts *o);
int dist_sample(Sampler *S, int exclude);
int is_stop(int id);
int keyed_pick(Sampler *S, const float *lg, int V, int pos);
int multi_accept(Sampler *S, const float *lg, int V, const int *cand, int nc, int *bonus);
int parse_ids(const char *s, int **out);
int parse_opts(int argc, char **argv, int i0, Opts *o);
void apply_quant(Q35State *s, const Opts *o);
int pld(const int *seq, int n, int nmax, int nmin, int gamma, int *out);
int pool_get(int key, int gamma, int *out);
int second_best(const float *lg, int V, int best);
int tr_chain(const TreeDraft *R, const int *y, int u, int *out, int maxlen);
void dist_build(Sampler *S, const float *lg, int V);
void json_top(FILE *f, int pos, int chosen, const float *lg, int V, int K);
void load_stops(const char *path);
void load_vocab(const char *path);
void load_vocab_once(const void *o);
void pool_add(int key, const int *cont, int len);
void print_token(FILE *f, int id);
void report(const Q35Model *m, const Q35State *s, double secs, int ntok);
void tr_build(TreeDraft *R, const int *seq, int n, int ngram, const int *orc, int norc);

#endif
