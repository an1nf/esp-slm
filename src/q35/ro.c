/* Read-only GDN pass: S0 = gamma B + sum w u k^T, materialized every R tokens. */
#include "q35_int.h"

static int is_hi(const Q35State *s, int l, int h, int Hv)
{
    return s->hi && s->hi[(size_t)l * Hv + h];
}

/* Store one v-head of S0: int8 rows (smoothed, dithered) or f16 for 16-bit heads. */
static void ro_quant_head(const Q35Model *m, Q35State *s, int l, int h, const float *src, int pos)
{
    const int Hv = m->c.v_heads, dk = m->c.d_k, dv = m->c.d_v;
    if (is_hi(s, l, h, Hv)) {
        float *B = s->ssm[l] + (size_t)h * dv * dk;
        if (B != src)
            memcpy(B, src, (size_t)dv * dk * sizeof(float));
        for (int j = 0; j < dv; j++)
            state_round(B + (size_t)j * dk, dk, Q35_STATE_F16);
        return;
    }
    float *cs = s->ro_sm[l] + (size_t)h * dk;
    if (s->smooth)
        q8_smooth(src, dv, dk, cs);
    else
        for (int i = 0; i < dk; i++)
            cs[i] = 1.0f;
    for (int j = 0; j < dv; j++) {
        const size_t r = (size_t)h * dv + j;
        s->ro_bs[l][r] = q8_row(src + (size_t)j * dk, dk, s->smooth ? cs : NULL,
                                q8_phase(l, h, j, pos), s->dither, s->ro_b[l] + r * dk);
    }
}

static int ro_alloc(const Q35Model *m, Q35State *s)
{
    const Q35Config *c = &m->c;
    const int L = c->n_layer, Hv = c->v_heads, dk = c->d_k, dv = c->d_v;
    const int kd = c->k_heads * dk, vd = Hv * dv;
    if (s->ro_n)
        return 0;
    if (s->ro_rmax <= 0)
        s->ro_rmax = 16;
    s->ro_cap = s->ro_rmax + s->max_T;
    const size_t cap = (size_t)s->ro_cap, T = (size_t)s->max_T;
    s->ro_n = (int *)calloc((size_t)L, sizeof(int));
    s->ro_b = (int8_t **)calloc((size_t)L, sizeof(int8_t *));
    float ***arrs[] = { &s->ro_bs, &s->ro_lgam, &s->ro_U, &s->ro_K, &s->ro_w, &s->ro_ut, &s->ro_lG };
    for (size_t a = 0; a < sizeof(arrs) / sizeof(arrs[0]); a++)
        *arrs[a] = (float **)calloc((size_t)L, sizeof(float *));
    s->ro_sm = (float **)calloc((size_t)L, sizeof(float *));
    s->ro_tmp = (float *)calloc((size_t)dv * dk, 4);
    if (!s->ro_sm || !s->ro_tmp || (s->hi_frac > 0.0f && q35_pick_hi(m, s)))
        return -1;
    s->ro_q = (float *)calloc(T * (size_t)kd, 4);
    s->ro_ak = (float *)calloc(T * (size_t)dv, 4);
    s->ro_aq = (float *)calloc(T * (size_t)dv, 4);
    if (!s->ro_n || !s->ro_b || !s->ro_q || !s->ro_ak || !s->ro_aq)
        return -1;
    for (int l = 0; l < L; l++) {
        if (!m->L[l].recurrent)
            continue;
        s->ro_b[l] = (int8_t *)calloc((size_t)Hv * dv * dk, 1);
        s->ro_bs[l] = (float *)calloc((size_t)Hv * dv, 4);
        s->ro_lgam[l] = (float *)calloc((size_t)Hv, 4);
        s->ro_U[l] = (float *)calloc(cap * (size_t)vd, 4);
        s->ro_K[l] = (float *)calloc(cap * (size_t)Hv * dk, 4);
        s->ro_w[l] = (float *)calloc(cap * (size_t)Hv, 4);
        s->ro_ut[l] = (float *)calloc(T * (size_t)vd, 4);
        s->ro_lG[l] = (float *)calloc(T * (size_t)Hv, 4);
        s->ro_sm[l] = (float *)calloc((size_t)Hv * dk, 4);
        if (!s->ro_b[l] || !s->ro_bs[l] || !s->ro_lgam[l] || !s->ro_U[l] || !s->ro_K[l]
            || !s->ro_w[l] || !s->ro_ut[l] || !s->ro_lG[l] || !s->ro_sm[l])
            return -1;
        for (int i = 0; i < Hv * dk; i++)
            s->ro_sm[l][i] = 1.0f;
        if (s->state_fmt == Q35_STATE_RO8)
            for (int h = 0; h < Hv; h++)
                ro_quant_head(m, s, l, h, s->ssm[l] + (size_t)h * dv * dk, s->n_past);
    }
    (void)kd;
    return 0;
}

void ro_free(Q35State *s, int L)
{
    if (!s->ro_n)
        return;
    for (int l = 0; l < L; l++) {
        if (s->ro_b) free(s->ro_b[l]);
        float **arrs[] = { s->ro_bs, s->ro_lgam, s->ro_U, s->ro_K, s->ro_w, s->ro_ut, s->ro_lG,
                           s->ro_sm };
        for (size_t a = 0; a < sizeof(arrs) / sizeof(arrs[0]); a++)
            if (arrs[a])
                free(arrs[a][l]);
    }
    free(s->ro_b); free(s->ro_bs); free(s->ro_lgam); free(s->ro_U); free(s->ro_K);
    free(s->ro_w); free(s->ro_ut); free(s->ro_lG); free(s->ro_n);
    free(s->ro_q); free(s->ro_ak); free(s->ro_aq); free(s->ro_sm); free(s->ro_tmp);
    s->ro_sm = NULL;
    s->ro_tmp = NULL;
    s->ro_n = NULL;
}

/* Fold the rank-R delta into the base and requantize. */
static void ro_materialize(const Q35Model *m, Q35State *s, int l)
{
    const Q35Config *c = &m->c;
    const int Hv = c->v_heads, dk = c->d_k, dv = c->d_v, n = s->ro_n[l];
    const int q8 = s->state_fmt == Q35_STATE_RO8;
    float *tmp = s->ro_tmp;
    for (int h = 0; h < Hv; h++) {
        const float gam = EXPF(s->ro_lgam[l][h]);
        const int hi = is_hi(s, l, h, Hv);
        const float *cs = s->ro_sm[l] + (size_t)h * dk;
        for (int j = 0; j < dv; j++) {
            const size_t r = (size_t)h * dv + j;
            float *row = tmp + (size_t)j * dk;
            if (q8 && !hi) {
                const float sc = s->ro_bs[l][r] * gam;
                for (int i = 0; i < dk; i++)
                    row[i] = (float)s->ro_b[l][r * dk + i] * sc * cs[i];
            } else {
                const float *B = s->ssm[l] + r * dk;
                for (int i = 0; i < dk; i++)
                    row[i] = B[i] * gam;
            }
            for (int e = 0; e < n; e++) {
                const float om = EXPF(s->ro_w[l][(size_t)e * Hv + h]);
                const float uj = s->ro_U[l][((size_t)e * Hv + h) * dv + j] * om;
                const float *kk = s->ro_K[l] + ((size_t)e * Hv + h) * dk;
                for (int i = 0; i < dk; i++)
                    row[i] += uj * kk[i];
            }
        }
        if (q8) {
            ro_quant_head(m, s, l, h, tmp, s->n_past);
        } else {
            if (s->state_fmt == Q35_STATE_RO16)
                for (int j = 0; j < dv; j++)
                    state_round(tmp + (size_t)j * dk, dk, Q35_STATE_F16);
            memcpy(s->ssm[l] + (size_t)h * dv * dk, tmp, (size_t)dv * dk * sizeof(float));
        }
        s->ro_lgam[l][h] = 0.0f;
    }
    s->ro_n[l] = 0;
    s->ro_mat++;
}

/* Append committed tokens as (u, k, log decay) terms of the delta. */
static void ro_commit(const Q35Model *m, Q35State *s, int l, int n)
{
    const Q35Config *c = &m->c;
    const int Hk = c->k_heads, Hv = c->v_heads, dk = c->d_k, dv = c->d_v;
    const int cd = conv_dim(c), K = c->d_conv, vd = Hv * dv;
    const size_t st = log_stride(c);
    float *cs = s->conv[l];
    for (int e = 0; e < n; e++) {
        const float *in = s->lg[l] + (size_t)s->pend[e] * st;
        if (K > 2)
            memmove(cs, cs + cd, (size_t)(K - 2) * cd * sizeof(float));
        memcpy(cs + (size_t)(K - 2) * cd, in, (size_t)cd * sizeof(float));
    }
    if (n <= 0)
        return;
    const int last = s->pend[n - 1];
    for (int h = 0; h < Hv; h++) {
        const float LG = s->ro_lG[l][(size_t)last * Hv + h];
        s->ro_lgam[l][h] += LG;
        for (int e = 0; e < s->ro_n[l]; e++)
            s->ro_w[l][(size_t)e * Hv + h] += LG;
    }
    for (int e = 0; e < n; e++) {
        const int node = s->pend[e];
        const int idx = s->ro_n[l]++;
        const float *kn = s->lg[l] + (size_t)node * st + cd;
        memcpy(s->ro_U[l] + (size_t)idx * vd, s->ro_ut[l] + (size_t)node * vd,
               (size_t)vd * sizeof(float));
        for (int h = 0; h < Hv; h++) {
            memcpy(s->ro_K[l] + ((size_t)idx * Hv + h) * dk, kn + (size_t)(h % Hk) * dk,
                   (size_t)dk * sizeof(float));
            s->ro_w[l][(size_t)idx * Hv + h] = s->ro_lG[l][(size_t)last * Hv + h]
                                             - s->ro_lG[l][(size_t)node * Hv + h];
        }
    }
    if (s->ro_n[l] > s->ro_rmax)
        ro_materialize(m, s, l);
}

/* y = S0 x with S0 = gamma B + sum_e w_e u_e k_e^T; int8 B is read against int8 or int16 x. */
static void ro_apply(const Q35Model *m, const Q35State *s, int l, int h, const float *x, float *y)
{
    const Q35Config *c = &m->c;
    const int Hv = c->v_heads, dk = c->d_k, dv = c->d_v;
    const float gam = EXPF(s->ro_lgam[l][h]);
    if (s->state_fmt == Q35_STATE_RO8 && !is_hi(s, l, h, Hv)) {
        float xs[512], xd[16];
        int8_t xq[512];
        int16_t xw[512];
        const float *cs = s->ro_sm[l] + (size_t)h * dk;
        for (int i = 0; i < dk; i++)
            xs[i] = s->smooth ? x[i] * cs[i] : x[i];
        const int nb = dk / 32;
        for (int b = 0; b < nb; b++) {
            if (s->x16) {
                float amax = 0.0f;
                for (int e = 0; e < 32; e++)
                    if (fabsf(xs[b * 32 + e]) > amax)
                        amax = fabsf(xs[b * 32 + e]);
                xd[b] = amax / 32767.0f;
                const float id = amax > 0.0f ? 32767.0f / amax : 0.0f;
                for (int e = 0; e < 32; e++)
                    xw[b * 32 + e] = (int16_t)rintf(xs[b * 32 + e] * id);
            } else {
                q35_quant_q8(xs + b * 32, 32, xq + b * 32, &xd[b]);
            }
        }
        const int8_t *B = s->ro_b[l] + (size_t)h * dv * dk;
        const float *bs = s->ro_bs[l] + (size_t)h * dv;
        for (int j = 0; j < dv; j++) {
            const int8_t *row = B + (size_t)j * dk;
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                int32_t si = 0;
                if (s->x16)
                    for (int e = 0; e < 32; e++)
                        si += (int32_t)row[b * 32 + e] * xw[b * 32 + e];
                else
                    for (int e = 0; e < 32; e++)
                        si += (int32_t)row[b * 32 + e] * xq[b * 32 + e];
                acc += xd[b] * (float)si;
            }
            y[j] = acc * bs[j] * gam;
        }
    } else {
        const float *B = s->ssm[l] + (size_t)h * dv * dk;
        for (int j = 0; j < dv; j++) {
            const float *row = B + (size_t)j * dk;
            float sum = 0.0f;
            for (int i = 0; i < dk; i++)
                sum += row[i] * x[i];
            y[j] = sum * gam;
        }
    }
    for (int e = 0; e < s->ro_n[l]; e++) {
        const float *kk = s->ro_K[l] + ((size_t)e * Hv + h) * dk;
        float kx = 0.0f;
        for (int i = 0; i < dk; i++)
            kx += kk[i] * x[i];
        const float f = kx * EXPF(s->ro_w[l][(size_t)e * Hv + h]);
        const float *u = s->ro_U[l] + ((size_t)e * Hv + h) * dv;
        for (int j = 0; j < dv; j++)
            y[j] += f * u[j];
    }
}

static float dotf(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

/* GDN pass that only reads S0: all nodes via S0 [k|q] plus a triangular solve along tree paths. */
void gdn_ro(const Q35Model *m, Q35State *s, int l, int T)
{
    const Q35Config *c = &m->c;
    const Q35Layer *Ly = &m->L[l];
    const int Hk = c->k_heads, Hv = c->v_heads;
    const int dk = c->d_k, dv = c->d_v, kd = Hk * dk, vd = Hv * dv;
    const int cd = conv_dim(c), K = c->d_conv;
    if (ro_alloc(m, s))
        return;
    ro_commit(m, s, l, s->n_pend);
    const size_t cw = (size_t)(K - 1) * cd, st = log_stride(c);
    const float scale = 1.0f / sqrtf((float)dv);
    float *cs = s->C_tmp;
    if (!s->tree)
        memcpy(s->C_tmp, s->conv[l], cw * sizeof(float));

    for (int t = 0; t < T; t++) {
        const float *in = s->big + (size_t)t * cd;
        if (s->tree) {
            const int d = s->tdep[t];
            const float *pc = d ? s->C_stk + (size_t)(d - 1) * cw : s->conv[l];
            cs = s->C_stk + (size_t)d * cw;
            memcpy(cs, pc, cw * sizeof(float));
        }
        float *co = s->co;
        float *lgt = s->lg[l] + (size_t)t * st;
        memcpy(lgt, in, (size_t)cd * sizeof(float));
        for (int ch = 0; ch < cd; ch++) {
            const float *w = Ly->conv + (size_t)ch * K;
            float sum = 0.0f;
            for (int k = 0; k < K - 1; k++)
                sum += cs[(size_t)k * cd + ch] * w[k];
            sum += in[ch] * w[K - 1];
            co[ch] = silu(sum);
        }
        if (K > 2)
            memmove(cs, cs + cd, (size_t)(K - 2) * cd * sizeof(float));
        memcpy(cs + (size_t)(K - 2) * cd, in, (size_t)cd * sizeof(float));
        for (int h = 0; h < 2 * Hk; h++)
            l2norm(co + (size_t)h * dk, dk, c->eps);
        float *lk = lgt + cd, *lv = lk + kd, *lb = lv + vd, *le = lb + Hv;
        memcpy(s->ro_q + (size_t)t * kd, co, (size_t)kd * sizeof(float));
        memcpy(lk, co + kd, (size_t)kd * sizeof(float));
        memcpy(lv, co + 2 * kd, (size_t)vd * sizeof(float));
        for (int h = 0; h < Hv; h++) {
            const float xa = s->al[(size_t)t * Hv + h] + Ly->ssm_dt[h];
            const float sp = xa > 20.0f ? xa : LOGF(1.0f + EXPF(xa));
            lb[h] = sigm(s->bt[(size_t)t * Hv + h]);
            le[h] = Ly->ssm_a[h] * sp;
        }
    }

    float *ut = s->ro_ut[l], *lG = s->ro_lG[l];
    for (int h = 0; h < Hv; h++) {
        const int hk = h % Hk;
        for (int t = 0; t < T; t++) {
            const float *lgt = s->lg[l] + (size_t)t * st;
            ro_apply(m, s, l, h, lgt + cd + (size_t)hk * dk, s->ro_ak + (size_t)t * dv);
            ro_apply(m, s, l, h, s->ro_q + (size_t)t * kd + (size_t)hk * dk,
                     s->ro_aq + (size_t)t * dv);
        }
        for (int t = 0; t < T; t++) {
            const float *lgt = s->lg[l] + (size_t)t * st;
            const float *kt = lgt + cd + (size_t)hk * dk;
            const float *vt = lgt + cd + kd + (size_t)h * dv;
            const float beta = lgt[cd + kd + vd + h];
            const float *qt = s->ro_q + (size_t)t * kd + (size_t)hk * dk;
            const int par = s->tree ? s->tpar[t] : t - 1;
            const float lGt = (par < 0 ? 0.0f : lG[(size_t)par * Hv + h]) + lgt[cd + kd + vd + Hv + h];
            lG[(size_t)t * Hv + h] = lGt;
            const float Gt = EXPF(lGt);
            float *u = ut + (size_t)t * vd + (size_t)h * dv;
            const float *ak = s->ro_ak + (size_t)t * dv, *aq = s->ro_aq + (size_t)t * dv;
            float *oh = s->att + (size_t)t * vd + (size_t)h * dv;
            for (int j = 0; j < dv; j++) {
                u[j] = vt[j] - Gt * ak[j];
                oh[j] = Gt * aq[j];
            }
            for (int i = par; i >= 0; i = s->tree ? s->tpar[i] : i - 1) {
                const float *ki = s->lg[l] + (size_t)i * st + cd + (size_t)hk * dk;
                const float r = EXPF(lGt - lG[(size_t)i * Hv + h]);
                const float fk = r * dotf(ki, kt, dk), fq = r * dotf(ki, qt, dk);
                const float *ui = ut + (size_t)i * vd + (size_t)h * dv;
                for (int j = 0; j < dv; j++) {
                    u[j] -= fk * ui[j];
                    oh[j] += fq * ui[j];
                }
            }
            for (int j = 0; j < dv; j++)
                u[j] *= beta;
            const float fq = dotf(kt, qt, dk);
            for (int j = 0; j < dv; j++)
                oh[j] = (oh[j] + fq * u[j]) * scale;
        }
    }

    for (int t = 0; t < T; t++)
        for (int h = 0; h < Hv; h++) {
            float *oh = s->att + (size_t)t * vd + (size_t)h * dv;
            rmsnorm(oh, oh, Ly->ssm_norm, dv, c->eps);
            const float *zh = s->z + (size_t)t * vd + (size_t)h * dv;
            for (int j = 0; j < dv; j++)
                oh[j] = silu(zh[j]) * oh[j];
        }
}
