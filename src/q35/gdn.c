/* Gated DeltaNet layer: chain and tree passes, deferred state commit. */
#include "q35_int.h"

void state_round(float *x, int n, int fmt)
{
    if (fmt == Q35_STATE_F16) {
        for (int i = 0; i < n; i++)
            x[i] = q35_f16_to_f32(q35_f32_to_f16(x[i]));
    } else if (fmt == Q35_STATE_BF16) {
        for (int i = 0; i < n; i++) {
            uint32_t u;
            memcpy(&u, &x[i], 4);
            if ((u & 0x7f800000u) != 0x7f800000u)
                u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
            memcpy(&x[i], &u, 4);
        }
    } else if (fmt == Q35_STATE_I8) {
        float amax = 0.0f;
        for (int i = 0; i < n; i++)
            if (fabsf(x[i]) > amax)
                amax = fabsf(x[i]);
        if (amax == 0.0f)
            return;
        const float d = amax / 127.0f, id = 127.0f / amax;
        for (int i = 0; i < n; i++)
            x[i] = rintf(x[i] * id) * d;
    }
}

int conv_dim(const Q35Config *c)
{
    return 2 * c->k_heads * c->d_k + c->v_heads * c->d_v;
}

size_t log_stride(const Q35Config *c)
{
    return (size_t)conv_dim(c) + (size_t)c->k_heads * c->d_k
           + (size_t)c->v_heads * c->d_v + 2u * (size_t)c->v_heads;
}

/* S = g S + beta (v - g S k) k^T */
static void delta_step(float *M, const float *k, const float *v, float beta,
                       float eg, int dk, int dv, float *tmp)
{
    for (int i = 0; i < dv * dk; i++)
        M[i] *= eg;
    for (int j = 0; j < dv; j++) {
        const float *row = M + (size_t)j * dk;
        float sum = 0.0f;
        for (int i = 0; i < dk; i++)
            sum += row[i] * k[i];
        tmp[j] = (v[j] - sum) * beta;
    }
    for (int j = 0; j < dv; j++) {
        float *row = M + (size_t)j * dk;
        const float dj = tmp[j];
        for (int i = 0; i < dk; i++)
            row[i] += k[i] * dj;
    }
}

static void state_store(float *M, int dk, int dv, const Q35State *s)
{
    if (s->state_fmt != Q35_STATE_F32)
        for (int j = 0; j < dv; j++)
            state_round(M + (size_t)j * dk, dk, s->state_fmt);
    else if (s->noise_eps > 0.0f) {
        const size_t n = (size_t)dk * dv;
        for (size_t i = 0; i < n; i++) {
            uint32_t b;
            memcpy(&b, &M[i], 4);
            uint32_t x = b ^ (uint32_t)(i * 2654435761u);
            x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
            M[i] *= 1.0f + ((x & 1) ? s->noise_eps : -s->noise_eps);
        }
    }
}

/* Apply committed tokens from the per-pass log with the same rounding as the pass. */
static void gdn_replay(const Q35Model *m, Q35State *s, int l, int n)
{
    const Q35Config *c = &m->c;
    const int Hk = c->k_heads, Hv = c->v_heads, dk = c->d_k, dv = c->d_v;
    const int cd = conv_dim(c), kd = Hk * dk, vd = Hv * dv, K = c->d_conv;
    const size_t st = log_stride(c);
    float *cs = s->conv[l], *S = s->ssm[l];
    for (int e = 0; e < n; e++) {
        const float *in = s->lg[l] + (size_t)s->pend[e] * st;
        const float *k = in + cd, *v = k + kd, *bt = v + vd, *eg = bt + Hv;
        if (K > 2)
            memmove(cs, cs + cd, (size_t)(K - 2) * cd * sizeof(float));
        memcpy(cs + (size_t)(K - 2) * cd, in, (size_t)cd * sizeof(float));
        for (int h = 0; h < Hv; h++) {
            float *M = S + (size_t)h * dv * dk;
            delta_step(M, k + (size_t)(h % Hk) * dk, v + (size_t)h * dv, bt[h], eg[h],
                       dk, dv, s->rc);
            state_store(M, dk, dv, s);
        }
    }
}

void gdn(const Q35Model *m, Q35State *s, int l, int T)
{
    const Q35Config *c = &m->c;
    const Q35Layer *Ly = &m->L[l];
    const int E = c->n_embd, Hk = c->k_heads, Hv = c->v_heads;
    const int dk = c->d_k, dv = c->d_v, kd = Hk * dk, vd = Hv * dv;
    const int cd = conv_dim(c), K = c->d_conv;
    int qr = 0;
    linear(&Ly->wqkv, s, T, s->xn, E, &qr, s->big, cd);
    linear(&Ly->wz, s, T, s->xn, E, &qr, s->z, vd);
    linear(&Ly->beta, s, T, s->xn, E, &qr, s->bt, Hv);
    linear(&Ly->alpha, s, T, s->xn, E, &qr, s->al, Hv);
    if (s->state_fmt == Q35_STATE_RO32 || s->state_fmt == Q35_STATE_RO8
        || s->state_fmt == Q35_STATE_RO16) {
        gdn_ro(m, s, l, T);
        int q2 = 0;
        linear(&Ly->ssm_out, s, T, s->att, vd, &q2, s->o, E);
        return;
    }

    if (s->n_pend)
        gdn_replay(m, s, l, s->n_pend);
    const size_t ss = (size_t)Hv * dv * dk, cw = (size_t)(K - 1) * cd;
    float *cs = s->C_tmp;
    float *S = s->S_tmp;
    if (!s->tree) {
        memcpy(s->C_tmp, s->conv[l], cw * sizeof(float));
        memcpy(s->S_tmp, s->ssm[l], ss * sizeof(float));
    }
    const size_t st = log_stride(c);
    const float scale = 1.0f / sqrtf((float)dv);
    for (int t = 0; t < T; t++) {
        const float *in = s->big + (size_t)t * cd;
        if (s->tree) {
            const int d = s->tdep[t];
            const float *ps = d ? s->S_stk + (size_t)(d - 1) * ss : s->ssm[l];
            const float *pc = d ? s->C_stk + (size_t)(d - 1) * cw : s->conv[l];
            S = s->S_stk + (size_t)d * ss;
            cs = s->C_stk + (size_t)d * cw;
            memcpy(S, ps, ss * sizeof(float));
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
        const float *qv = co, *kv = co + kd, *vv = co + 2 * kd;
        float *lk = lgt + cd, *lv = lk + kd, *lb = lv + vd, *le = lb + Hv;
        memcpy(lk, kv, (size_t)kd * sizeof(float));
        memcpy(lv, vv, (size_t)vd * sizeof(float));
        float *out = s->att + (size_t)t * vd;
        for (int h = 0; h < Hv; h++) {
            const int hk = h % Hk;
            const float *q = qv + (size_t)hk * dk, *k = kv + (size_t)hk * dk;
            const float *v = vv + (size_t)h * dv;
            const float beta = sigm(s->bt[(size_t)t * Hv + h]);
            const float xa = s->al[(size_t)t * Hv + h] + Ly->ssm_dt[h];
            const float sp = xa > 20.0f ? xa : LOGF(1.0f + EXPF(xa));
            const float eg = EXPF(Ly->ssm_a[h] * sp);
            lb[h] = beta;
            le[h] = eg;
            float *M = S + (size_t)h * dv * dk;
            delta_step(M, k, v, beta, eg, dk, dv, s->rc);
            float *oh = out + (size_t)h * dv;
            for (int j = 0; j < dv; j++) {
                const float *row = M + (size_t)j * dk;
                float sum = 0.0f;
                for (int i = 0; i < dk; i++)
                    sum += row[i] * q[i];
                oh[j] = sum * scale;
            }
            state_store(M, dk, dv, s);

            rmsnorm(oh, oh, Ly->ssm_norm, dv, c->eps);
            const float *zh = s->z + (size_t)t * vd + (size_t)h * dv;
            for (int j = 0; j < dv; j++)
                oh[j] = silu(zh[j]) * oh[j];
        }
    }
    int q2 = 0;
    linear(&Ly->ssm_out, s, T, s->att, vd, &q2, s->o, E);
}
