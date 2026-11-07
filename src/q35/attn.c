/* Gated attention with partial RoPE; f16/f32 KV, optional low-bit KV with exact top-R rows. */
#include "q35_int.h"

static void rope_cache(const Q35Config *c, const int sect[4], int pos,
                       float *rc, float *rs)
{
    const int half = c->n_rot / 2;
    const float ts = POWF(c->rope_base, -2.0f / (float)c->n_rot);
    const int sd = sect[0] + sect[1] + sect[2] + sect[3];
    float th = (float)pos, te = 0.0f;
    for (int i = 0; i < half; i++) {
        const int sec = i % sd;
        int is_e = 1;
        if (sec % 3 == 1 && sec < 3 * sect[1])
            is_e = 0;
        else if (sec % 3 == 2 && sec < 3 * sect[2])
            is_e = 0;
        else if (sec % 3 == 0 && sec < 3 * sect[0])
            is_e = 0;
        const float a = is_e ? te : th;
        rc[i] = COSF(a);
        rs[i] = SINF(a);
        th *= ts;
        te *= ts;
    }
}

static void rope_apply(float *v, int half, const float *rc, const float *rs)
{
    for (int i = 0; i < half; i++) {
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * rc[i] - x1 * rs[i];
        v[i + half] = x0 * rs[i] + x1 * rc[i];
    }
}

static int kv_alloc(const Q35Model *m, Q35State *s)
{
    const Q35Config *c = &m->c;
    const size_t L = (size_t)c->n_layer, kvd = (size_t)c->n_kv * c->head_dim;
    const size_t n = (size_t)s->max_seq * kvd;
    s->kq = (int8_t **)calloc(L, sizeof(int8_t *));
    s->vq = (int8_t **)calloc(L, sizeof(int8_t *));
    s->ks = (uint16_t **)calloc(L, sizeof(uint16_t *));
    s->vs = (uint16_t **)calloc(L, sizeof(uint16_t *));
    s->kv_sel = (float *)calloc((size_t)s->max_seq + 1, 4);
    s->kv_deq = (float *)calloc((size_t)c->head_dim, 4);
    s->kv_ex = (uint8_t *)calloc((size_t)s->max_seq + 1, 1);
    if (!s->kq || !s->vq || !s->ks || !s->vs || !s->kv_sel || !s->kv_deq || !s->kv_ex)
        return -1;
    for (size_t l = 0; l < L; l++) {
        if (m->L[l].recurrent)
            continue;
        s->kq[l] = (int8_t *)calloc(n, 1);
        s->vq[l] = (int8_t *)calloc(n, 1);
        s->ks[l] = (uint16_t *)calloc(n / 32, 2);
        s->vs[l] = (uint16_t *)calloc(n / 32, 2);
        if (!s->kq[l] || !s->vq[l] || !s->ks[l] || !s->vs[l])
            return -1;
    }
    return 0;
}

/* k-th largest of x[0..n) (x is reordered). */
static float kth_largest(float *x, int n, int k)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const float piv = x[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (x[i] > piv)
                i++;
            while (x[j] < piv)
                j--;
            if (i <= j) {
                const float t = x[i];
                x[i] = x[j];
                x[j] = t;
                i++;
                j--;
            }
        }
        if (k <= j)
            hi = j;
        else if (k >= i)
            lo = i;
        else
            break;
    }
    return x[k];
}

static float dot_full(const Q35State *s, int l, size_t o, const float *q, int hd)
{
    float sum = 0.0f;
    if (s->kv_f16) {
        const uint16_t *kr = (const uint16_t *)s->kc[l] + o;
        for (int i = 0; i < hd; i++)
            sum += q[i] * q35_f16_to_f32(kr[i]);
    } else {
        const float *kr = (const float *)s->kc[l] + o;
        for (int i = 0; i < hd; i++)
            sum += q[i] * kr[i];
    }
    return sum;
}

void attn(const Q35Model *m, Q35State *s, int l, int T)
{
    const Q35Config *c = &m->c;
    const Q35Layer *Ly = &m->L[l];
    const int E = c->n_embd, nh = c->n_head, nkv = c->n_kv, hd = c->head_dim;
    const int qd = 2 * nh * hd, kvd = nkv * hd, half = c->n_rot / 2;
    int qr = 0;
    linear(&Ly->wq, s, T, s->xn, E, &qr, s->big, qd);
    linear(&Ly->wk, s, T, s->xn, E, &qr, s->kbuf, kvd);
    linear(&Ly->wv, s, T, s->xn, E, &qr, s->vbuf, kvd);
    const float scale = 1.0f / sqrtf((float)hd);
    const int grp = nh / nkv;
    const int lo = s->kv_bits > 0;
    if (lo && !s->kq && kv_alloc(m, s))
        return;
    for (int t = 0; t < T; t++) {
        const int pos = s->n_past + (s->tree ? s->tdep[t] : t);
        rope_cache(c, c->sect, pos, s->rc, s->rs);
        float *qg = s->big + (size_t)t * qd;
        for (int h = 0; h < nh; h++) {
            float *q = qg + (size_t)h * 2 * hd;
            rmsnorm(q, q, Ly->q_norm, hd, c->eps);
            rope_apply(q, half, s->rc, s->rs);
        }
        float *k = s->kbuf + (size_t)t * kvd, *v = s->vbuf + (size_t)t * kvd;
        for (int h = 0; h < nkv; h++) {
            rmsnorm(k + (size_t)h * hd, k + (size_t)h * hd, Ly->k_norm, hd, c->eps);
            rope_apply(k + (size_t)h * hd, half, s->rc, s->rs);
        }
        const size_t off = (size_t)(s->n_past + t) * kvd;
        if (s->kv_f16) {
            uint16_t *kc = (uint16_t *)s->kc[l] + off, *vc = (uint16_t *)s->vc[l] + off;
            for (int i = 0; i < kvd; i++) {
                kc[i] = q35_f32_to_f16(k[i]);
                vc[i] = q35_f32_to_f16(v[i]);
            }
        } else {
            memcpy((float *)s->kc[l] + off, k, (size_t)kvd * sizeof(float));
            memcpy((float *)s->vc[l] + off, v, (size_t)kvd * sizeof(float));
        }
        if (lo) {
            kv_quant(k, kvd, s->kv_bits, s->kq[l] + off, s->ks[l] + off / 32);
            kv_quant(v, kvd, s->kv_bits, s->vq[l] + off, s->vs[l] + off / 32);
        }
    }
    for (int t = 0; t < T; t++) {
        int np = s->n_past;
        for (int p = 0; p < s->n_past; p++)
            s->slots[p] = p;
        if (s->tree) {
            const int d = s->tdep[t];
            for (int u = t, k = d; u >= 0; u = s->tpar[u], k--)
                s->slots[s->n_past + k] = s->n_past + u;
            np += d + 1;
        } else {
            for (int u = 0; u <= t; u++)
                s->slots[np++] = s->n_past + u;
        }
        const int pos = np - 1;
        const float *qg = s->big + (size_t)t * qd;
        float *out = s->att + (size_t)t * nh * hd;
        for (int h = 0; h < nh; h++) {
            const float *q = qg + (size_t)h * 2 * hd;
            const float *gate = q + hd;
            const int kh = h / grp;
            float mx = -INFINITY;
            if (lo) {
                int nold = 0;
                for (int p = 0; p <= pos; p++) {
                    const size_t o = (size_t)s->slots[p] * kvd + (size_t)kh * hd;
                    s->kv_ex[p] = pos - p < s->kv_recent;
                    if (s->kv_ex[p]) {
                        s->scores[p] = dot_full(s, l, o, q, hd) * scale;
                        continue;
                    }
                    kv_dequant(s->kq[l] + o, s->ks[l] + o / 32, hd, s->kv_bits, s->kv_deq);
                    float sum = 0.0f;
                    for (int i = 0; i < hd; i++)
                        sum += q[i] * s->kv_deq[i];
                    s->scores[p] = sum * scale;
                    s->kv_sel[nold++] = s->scores[p];
                }
                s->kv_rows += (uint64_t)nold;
                if (s->kv_refine > 0 && nold > 0) {
                    const int R = s->kv_refine < nold ? s->kv_refine : nold;
                    const float thr = kth_largest(s->kv_sel, nold, R - 1);
                    int left = R;
                    for (int p = 0; p <= pos && left > 0; p++)
                        if (!s->kv_ex[p] && s->scores[p] > thr) {
                            s->kv_ex[p] = 2;
                            left--;
                        }
                    for (int p = 0; p <= pos && left > 0; p++)
                        if (!s->kv_ex[p] && s->scores[p] == thr) {
                            s->kv_ex[p] = 2;
                            left--;
                        }
                    for (int p = 0; p <= pos; p++)
                        if (s->kv_ex[p] == 2) {
                            const size_t o = (size_t)s->slots[p] * kvd + (size_t)kh * hd;
                            s->scores[p] = dot_full(s, l, o, q, hd) * scale;
                            s->kv_exact++;
                        }
                }
                for (int p = 0; p <= pos; p++)
                    if (s->scores[p] > mx)
                        mx = s->scores[p];
            } else {
                for (int p = 0; p <= pos; p++) {
                    const size_t o = (size_t)s->slots[p] * kvd + (size_t)kh * hd;
                    s->scores[p] = dot_full(s, l, o, q, hd) * scale;
                    if (s->scores[p] > mx)
                        mx = s->scores[p];
                }
            }
            float tot = 0.0f;
            for (int p = 0; p <= pos; p++) {
                s->scores[p] = EXPF(s->scores[p] - mx);
                tot += s->scores[p];
            }
            const float inv = 1.0f / tot;
            float *oh = out + (size_t)h * hd;
            for (int i = 0; i < hd; i++)
                oh[i] = 0.0f;
            for (int p = 0; p <= pos; p++) {
                const float pr = s->scores[p] * inv;
                const size_t o = (size_t)s->slots[p] * kvd + (size_t)kh * hd;
                if (lo && !s->kv_ex[p]) {
                    kv_dequant(s->vq[l] + o, s->vs[l] + o / 32, hd, s->kv_bits, s->kv_deq);
                    for (int i = 0; i < hd; i++)
                        oh[i] += pr * s->kv_deq[i];
                } else if (s->kv_f16) {
                    const uint16_t *vr = (const uint16_t *)s->vc[l] + o;
                    for (int i = 0; i < hd; i++)
                        oh[i] += pr * q35_f16_to_f32(vr[i]);
                } else {
                    const float *vr = (const float *)s->vc[l] + o;
                    for (int i = 0; i < hd; i++)
                        oh[i] += pr * vr[i];
                }
            }
            for (int i = 0; i < hd; i++)
                oh[i] = oh[i] * sigm(gate[i]);
        }
    }
    int q2 = 0;
    linear(&Ly->wo, s, T, s->att, nh * hd, &q2, s->o, E);
}

void ffn(const Q35Model *m, Q35State *s, int l, int T)
{
    const Q35Layer *Ly = &m->L[l];
    const int E = m->c.n_embd, F = m->c.n_ff;
    int qr = 0;
    linear(&Ly->gate, s, T, s->xn, E, &qr, s->g, F);
    linear(&Ly->up, s, T, s->xn, E, &qr, s->u, F);
    for (size_t i = 0; i < (size_t)T * F; i++)
        s->g[i] = silu(s->g[i]) * s->u[i];
    int q2 = 0;
    linear(&Ly->down, s, T, s->g, F, &q2, s->o, E);
}
