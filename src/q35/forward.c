/* Forward pass, tree evaluation, commit and state allocation. */
#include "q35_int.h"

void q35_commit(Q35State *s, int n)
{
    if (n < 0)
        n = 0;
    if (n > s->last_T)
        n = s->last_T;
    for (int k = 0; k < n; k++)
        s->pend[k] = k;
    s->n_past += n;
    s->n_pend = n;
    s->last_T = 0;
    s->tree = 0;
}

int q35_commit_path(const Q35Model *m, Q35State *s, const int *path, int n)
{
    if (n < 0 || n > s->last_T)
        return -1;
    for (int k = 0; k < n; k++) {
        if (path[k] < 0 || path[k] >= s->last_T)
            return -1;
        const int want = k ? path[k - 1] : -1;
        const int par = s->tree ? s->tpar[path[k]] : path[k] - 1;
        if (par != want)
            return -1;
    }

    const Q35Config *c = &m->c;
    const size_t kvd = (size_t)c->n_kv * c->head_dim, es = s->kv_f16 ? 2 : 4;
    for (int l = 0; l < c->n_layer; l++) {
        if (m->L[l].recurrent)
            continue;
        for (int k = 0; k < n; k++)
            if (path[k] != k) {
                memcpy((char *)s->kc[l] + (size_t)(s->n_past + k) * kvd * es,
                       (char *)s->kc[l] + (size_t)(s->n_past + path[k]) * kvd * es, kvd * es);
                memcpy((char *)s->vc[l] + (size_t)(s->n_past + k) * kvd * es,
                       (char *)s->vc[l] + (size_t)(s->n_past + path[k]) * kvd * es, kvd * es);
                if (s->kq) {
                    const size_t d = (size_t)(s->n_past + k) * kvd, f = (size_t)(s->n_past + path[k]) * kvd;
                    memcpy(s->kq[l] + d, s->kq[l] + f, kvd);
                    memcpy(s->vq[l] + d, s->vq[l] + f, kvd);
                    memcpy(s->ks[l] + d / 32, s->ks[l] + f / 32, kvd / 32 * 2);
                    memcpy(s->vs[l] + d / 32, s->vs[l] + f / 32, kvd / 32 * 2);
                }
            }
    }
    for (int k = 0; k < n; k++)
        s->pend[k] = path[k];
    s->n_past += n;
    s->n_pend = n;
    s->last_T = 0;
    s->tree = 0;
    return 0;
}

static const float *eval_core(const Q35Model *m, Q35State *s, const int *tok, int T,
                              int all_logits);

const float *q35_eval_tree(const Q35Model *m, Q35State *s, const int *tok,
                           const int *parent, int N)
{
    if (s->last_T)
        q35_commit(s, 0);
    if (N < 1 || N > s->max_T || s->n_past + N > s->max_seq)
        return NULL;
    int chain = 1;
    for (int i = 0; i < N; i++) {
        if (parent[i] >= i || parent[i] < -1)
            return NULL;
        if (parent[i] >= 0) {
            int u = i - 1;
            while (u >= 0 && u != parent[i])
                u = s->tpar[u];
            if (u < 0)
                return NULL;
        }
        s->tpar[i] = parent[i];
        s->tdep[i] = parent[i] < 0 ? 0 : s->tdep[parent[i]] + 1;
        chain &= parent[i] == i - 1;
    }
    if (!chain && !s->S_stk) {
        const Q35Config *c = &m->c;
        s->S_stk = (float *)calloc((size_t)s->max_T * c->v_heads * c->d_v * c->d_k, 4);
        s->C_stk = (float *)calloc((size_t)s->max_T * (c->d_conv - 1) * conv_dim(c), 4);
        if (!s->S_stk || !s->C_stk)
            return NULL;
    }
    s->tree = !chain;
    return eval_core(m, s, tok, N, 1);
}

const float *q35_forward(const Q35Model *m, Q35State *s, const int *tok, int T,
                         int all_logits)
{
    const float *lg = q35_eval(m, s, tok, T, all_logits);
    if (lg)
        q35_commit(s, T);
    return lg;
}

const float *q35_eval(const Q35Model *m, Q35State *s, const int *tok, int T,
                      int all_logits)
{
    if (s->last_T)
        q35_commit(s, 0);
    if (T < 1 || T > s->max_T || s->n_past + T > s->max_seq)
        return NULL;
    s->tree = 0;
    return eval_core(m, s, tok, T, all_logits);
}

static const float *eval_core(const Q35Model *m, Q35State *s, const int *tok, int T,
                              int all_logits)
{
    const Q35Config *c = &m->c;
    const int E = c->n_embd;
    for (int t = 0; t < T; t++) {
        dequant_row(&m->embd, tok[t], s->x + (size_t)t * E, s->tile);
        if (m->embd_inv.block)
            rot_inverse(&m->embd_inv, s->x + (size_t)t * E, E);
        s->bytes_kind[Q35_K_EMBD] += m->embd.row_bytes;
    }
    for (int l = 0; l < c->n_layer; l++) {
        const Q35Layer *Ly = &m->L[l];
        for (int t = 0; t < T; t++)
            rmsnorm(s->xn + (size_t)t * E, s->x + (size_t)t * E, Ly->attn_norm, E, c->eps);
        if (Ly->recurrent)
            gdn(m, s, l, T);
        else
            attn(m, s, l, T);
        for (size_t i = 0; i < (size_t)T * E; i++)
            s->x[i] += s->o[i];
        for (int t = 0; t < T; t++)
            rmsnorm(s->xn + (size_t)t * E, s->x + (size_t)t * E, Ly->post_norm, E, c->eps);
        ffn(m, s, l, T);
        for (size_t i = 0; i < (size_t)T * E; i++)
            s->x[i] += s->o[i];
        if (s->dbg) {
            char name[64] = {0};
            const uint32_t ne[2] = {(uint32_t)E, (uint32_t)T};
            snprintf(name, sizeof(name), "l_out-%d", l);
            fwrite(name, 1, 64, s->dbg);
            fwrite(ne, 4, 2, s->dbg);
            fwrite(s->x, 4, (size_t)T * E, s->dbg);
        }
    }
    const int t0 = all_logits ? 0 : T - 1, nt = T - t0;
    for (int t = t0; t < T; t++)
        rmsnorm(s->xn + (size_t)(t - t0) * E, s->x + (size_t)t * E, m->out_norm, E, c->eps);
    int qr = 0;
    linear(&m->head, s, nt, s->xn, E, &qr, s->logits, c->vocab);
    if (s->hdump)
        fwrite(rot_sig(&m->head.rot) ? s->hx : s->xn, 4, (size_t)nt * E, s->hdump);
    s->n_pend = 0;
    s->last_T = T;
    return s->logits;
}

int q35_state_init(Q35State *s, const Q35Model *m, int max_seq, int max_T,
                   int kv_f16)
{
    const Q35Config *c = &m->c;
    memset(s, 0, sizeof(*s));
    s->max_seq = max_seq;
    s->max_T = max_T;
    s->kv_f16 = kv_f16;
    s->kv_recent = 16;
    const size_t T = (size_t)max_T, E = (size_t)c->n_embd;
    const size_t vd = (size_t)c->v_heads * c->d_v, cd = (size_t)conv_dim(c);
    const size_t qd = 2 * (size_t)c->n_head * c->head_dim, kvd = (size_t)c->n_kv * c->head_dim;
    const size_t ad = (size_t)c->n_head * c->head_dim;
    size_t mc = E;
    if ((size_t)c->n_ff > mc) mc = (size_t)c->n_ff;
    if (vd > mc) mc = vd;
    if (ad > mc) mc = ad;
    const size_t big = cd > qd ? cd : qd;
    s->x = (float *)calloc(T * E, 4);
    s->xn = (float *)calloc(T * E, 4);
    s->o = (float *)calloc(T * E, 4);
    s->big = (float *)calloc(T * big, 4);
    s->z = (float *)calloc(T * vd, 4);
    s->att = (float *)calloc(T * (vd > ad ? vd : ad), 4);
    s->kbuf = (float *)calloc(T * kvd, 4);
    s->vbuf = (float *)calloc(T * kvd, 4);
    s->g = (float *)calloc(T * (size_t)c->n_ff, 4);
    s->u = (float *)calloc(T * (size_t)c->n_ff, 4);
    s->bt = (float *)calloc(T * (size_t)c->v_heads, 4);
    s->al = (float *)calloc(T * (size_t)c->v_heads, 4);
    s->co = (float *)calloc(cd, 4);
    s->scores = (float *)calloc((size_t)max_seq, 4);
    s->logits = (float *)calloc(T * (size_t)c->vocab, 4);
    const size_t rr = (size_t)(c->d_v > c->n_rot ? c->d_v : c->n_rot);
    s->rc = (float *)calloc(rr, 4);
    s->rs = (float *)calloc(rr, 4);
    s->aq = (int8_t *)calloc(T * mc, 1);
    s->ad = (float *)calloc(T * mc / 32, 4);
    s->nthreads = 1;
#ifdef _OPENMP
    s->nthreads = omp_get_max_threads();
#endif
    s->tile_stride = mc;
    s->hx = (float *)calloc(T * mc, 4);
    if (!s->hx)
        return -1;
    s->tile = (int8_t *)calloc((size_t)s->nthreads * mc, 1);
    s->td = (float *)calloc((size_t)s->nthreads * (mc / 128 + 1), 4);
    s->conv = (float **)calloc((size_t)c->n_layer, sizeof(float *));
    s->ssm = (float **)calloc((size_t)c->n_layer, sizeof(float *));
    s->kc = (void **)calloc((size_t)c->n_layer, sizeof(void *));
    s->vc = (void **)calloc((size_t)c->n_layer, sizeof(void *));
    if (!s->x || !s->xn || !s->o || !s->big || !s->z || !s->att || !s->kbuf
        || !s->vbuf || !s->g || !s->u || !s->bt || !s->al || !s->co || !s->scores
        || !s->logits || !s->rc || !s->rs || !s->aq || !s->ad || !s->tile || !s->td
        || !s->conv || !s->ssm || !s->kc || !s->vc)
        return -1;
    const size_t es = kv_f16 ? 2 : 4;
    s->pend = (int *)calloc(T, sizeof(int));
    s->tpar = (int *)calloc(T, sizeof(int));
    s->tdep = (int *)calloc(T, sizeof(int));
    s->slots = (int *)calloc((size_t)max_seq + 1, sizeof(int));
    if (!s->pend || !s->tpar || !s->tdep || !s->slots)
        return -1;
    s->lg = (float **)calloc((size_t)c->n_layer, sizeof(float *));
    s->S_tmp = (float *)calloc((size_t)c->v_heads * c->d_v * c->d_k, 4);
    s->C_tmp = (float *)calloc((size_t)(c->d_conv - 1) * cd, 4);
    if (!s->lg || !s->S_tmp || !s->C_tmp)
        return -1;
    for (int l = 0; l < c->n_layer; l++) {
        if (m->L[l].recurrent) {
            s->lg[l] = (float *)calloc(T * log_stride(c), 4);
            if (!s->lg[l])
                return -1;
            s->conv[l] = (float *)calloc((size_t)(c->d_conv - 1) * cd, 4);
            s->ssm[l] = (float *)calloc((size_t)c->v_heads * c->d_v * c->d_k, 4);
            if (!s->conv[l] || !s->ssm[l])
                return -1;
        } else {
            s->kc[l] = calloc((size_t)max_seq * kvd, es);
            s->vc[l] = calloc((size_t)max_seq * kvd, es);
            if (!s->kc[l] || !s->vc[l])
                return -1;
        }
    }
    s->n_layers_alloc = c->n_layer;
    return 0;
}

void q35_reset(Q35State *s, const Q35Model *m)
{
    const Q35Config *c = &m->c;
    s->n_past = 0;
    s->n_pend = 0;
    s->last_T = 0;
    s->tree = 0;
    for (int l = 0; l < c->n_layer; l++)
        if (m->L[l].recurrent) {
            memset(s->conv[l], 0, (size_t)(c->d_conv - 1) * conv_dim(c) * 4);
            memset(s->ssm[l], 0, (size_t)c->v_heads * c->d_v * c->d_k * 4);
            if (s->ro_n) {
                s->ro_n[l] = 0;
                memset(s->ro_b[l], 0, (size_t)c->v_heads * c->d_v * c->d_k);
                memset(s->ro_bs[l], 0, (size_t)c->v_heads * c->d_v * 4);
                memset(s->ro_lgam[l], 0, (size_t)c->v_heads * 4);
                for (int i = 0; i < c->v_heads * c->d_k; i++)
                    s->ro_sm[l][i] = 1.0f;
            }
        }
}

void q35_state_free(Q35State *s)
{
    ro_free(s, s->n_layers_alloc);
    for (int l = 0; l < s->n_layers_alloc; l++) {
        if (s->lg) free(s->lg[l]);
        if (s->conv) free(s->conv[l]);
        if (s->ssm) free(s->ssm[l]);
        if (s->kc) free(s->kc[l]);
        if (s->vc) free(s->vc[l]);
        if (s->kq) {
            free(s->kq[l]);
            free(s->vq[l]);
            free(s->ks[l]);
            free(s->vs[l]);
        }
    }
    free(s->kq); free(s->vq); free(s->ks); free(s->vs);
    free(s->kv_sel); free(s->kv_deq); free(s->kv_ex); free(s->hi);
    free(s->conv); free(s->ssm); free(s->kc); free(s->vc); free(s->lg);
    free(s->S_tmp); free(s->C_tmp); free(s->S_stk); free(s->C_stk);
    free(s->pend); free(s->tpar); free(s->tdep); free(s->slots);
    free(s->x); free(s->xn); free(s->o); free(s->big); free(s->z); free(s->att);
    free(s->kbuf); free(s->vbuf); free(s->g); free(s->u); free(s->bt); free(s->al);
    free(s->co); free(s->scores); free(s->logits); free(s->rc); free(s->rs);
    free(s->aq); free(s->ad); free(s->tile); free(s->td); free(s->hx);
    memset(s, 0, sizeof(*s));
}
