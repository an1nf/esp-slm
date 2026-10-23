/* GGUF loading, PrismML Hadamard folding and config. */
#include "q35_int.h"

static void *keep(Q35Model *m, size_t n)
{
    if (m->nblobs == m->capblobs) {
        const int cap = m->capblobs ? 2 * m->capblobs : 256;
        void **nb = (void **)realloc(m->blobs, (size_t)cap * sizeof(void *));
        if (!nb)
            return NULL;
        m->blobs = nb;
        m->capblobs = cap;
    }
    void *p = malloc(n ? n : 1);
    if (p)
        m->blobs[m->nblobs++] = p;
    return p;
}

typedef struct {
    int version, tied, block, grouped, nnames, ninv, nw;
    char **names, **inv;
    int *used;
    int widths[32];
    float *signs[32];
} HdCfg;

typedef struct {
    Q35Model *m;
    GGUF *g;
    char *err;
    size_t errlen;
    HdCfg *hd;
} Ld;

static const float *hd_signs(const HdCfg *h, int width, int *ok)
{
    *ok = 1;
    if (h->nw == 0)
        return NULL;
    for (int i = 0; i < h->nw; i++)
        if (h->widths[i] == width)
            return h->signs[i];
    *ok = 0;
    return NULL;
}

static int ld_fail(Ld *L, const char *fmt, const char *a);

static int hd_assign(Ld *L, const char *name, Q35Mat *M)
{
    HdCfg *h = L->hd;
    if (!h || !h->version)
        return 0;
    for (int i = 0; i < h->nnames; i++) {
        if (strcmp(h->names[i], name))
            continue;
        int ok;
        memset(&M->rot, 0, sizeof(M->rot));
        M->rot.block = h->block;
        M->rot.signs = hd_signs(h, M->cols, &ok);
        if (!ok || M->cols % h->block)
            return ld_fail(L, "prism.hadamard: missing signs or block mismatch for %s", name);
        if (h->grouped && strstr(name, ".ssm_out.")) {
            const Q35Config *c = &L->m->c;
            M->rot.perm_hd = M->cols / c->v_heads;
            M->rot.perm_nk = c->k_heads;
            M->rot.perm_rep = c->v_heads / c->k_heads;
        }
        h->used[i] = 1;
        L->m->hd_folded++;
        return 0;
    }
    return 0;
}

static int hd_load(Ld *L, HdCfg *h)
{
    GGUF *g = L->g;
    uint32_t v = 0;
    memset(h, 0, sizeof(*h));
    if (gguf_u32(g, "prism.hadamard.version", &v))
        return 0;
    if (v != 1 && v != 2)
        return ld_fail(L, "prism.hadamard.version unsupported (%s)", v == 0 ? "0" : ">2");
    h->version = (int)v;
    uint32_t u = 0;
    if (!gguf_u32(g, "prism.hadamard.tied_output", &u))
        h->tied = u != 0;
    if (!gguf_u32(g, "prism.hadamard.gdn_v_grouped", &u))
        h->grouped = u != 0;
    if (gguf_u32(g, "prism.hadamard.block_size", &u) || !u || (u & (u - 1)))
        return ld_fail(L, "%s", "prism.hadamard.block_size invalid");
    h->block = (int)u;
    const char *tr = gguf_str(g, "prism.hadamard.transform");
    const char *ax = gguf_str(g, "prism.hadamard.axis");
    const char *sm = gguf_str(g, "prism.hadamard.sign_mode");
    if (!tr || strcmp(tr, "normalized-sylvester-walsh-hadamard"))
        return ld_fail(L, "prism.hadamard.transform unsupported: %s", tr ? tr : "?");
    if (!ax || strcmp(ax, "input-last-dimension"))
        return ld_fail(L, "prism.hadamard.axis unsupported: %s", ax ? ax : "?");
    if (!sm || (strcmp(sm, "identity") && strcmp(sm, "explicit")))
        return ld_fail(L, "prism.hadamard.sign_mode unsupported: %s", sm ? sm : "?");
    h->nnames = gguf_arr_str(g, "prism.hadamard.weight_names", &h->names);
    if (h->nnames < 1)
        return ld_fail(L, "%s", "prism.hadamard.weight_names empty");
    h->ninv = gguf_arr_str(g, "prism.hadamard.inverse_weight_names", &h->inv);
    if (h->ninv < 0)
        h->ninv = 0;
    h->used = (int *)calloc((size_t)h->nnames, sizeof(int));
    if (!h->used)
        return ld_fail(L, "%s", "out of memory");
    if (!strcmp(sm, "explicit")) {
        const GGUFKV *kw = gguf_kv(g, "prism.hadamard.sign_widths");
        const GGUFKV *kv = gguf_kv(g, "prism.hadamard.sign_values");
        if (!kw || !kv || kw->arr_n < 1 || kw->arr_n > 32)
            return ld_fail(L, "%s", "prism.hadamard: explicit signs without widths");
        h->nw = (int)kw->arr_n;
        if (gguf_arr_i32(g, "prism.hadamard.sign_widths", h->widths, h->nw) != h->nw)
            return ld_fail(L, "%s", "prism.hadamard.sign_widths unreadable");
        int32_t *vals = (int32_t *)malloc((size_t)kv->arr_n * sizeof(int32_t));
        if (!vals || gguf_arr_i32(g, "prism.hadamard.sign_values", vals, (int)kv->arr_n) != (int)kv->arr_n) {
            free(vals);
            return ld_fail(L, "%s", "prism.hadamard.sign_values unreadable");
        }
        size_t off = 0;
        for (int i = 0; i < h->nw; i++) {
            const int w = h->widths[i];
            if (w <= 0 || w % h->block || off + (size_t)w > kv->arr_n) {
                free(vals);
                return ld_fail(L, "%s", "prism.hadamard: invalid sign width");
            }
            float *sg = (float *)keep(L->m, (size_t)w * sizeof(float));
            if (!sg) {
                free(vals);
                return ld_fail(L, "%s", "out of memory");
            }
            for (int j = 0; j < w; j++) {
                if (vals[off + j] != 1 && vals[off + j] != -1) {
                    free(vals);
                    return ld_fail(L, "%s", "prism.hadamard: signs must be +-1");
                }
                sg[j] = (float)vals[off + j];
            }
            h->signs[i] = sg;
            off += (size_t)w;
        }
        free(vals);
        if (off != kv->arr_n)
            return ld_fail(L, "%s", "prism.hadamard.sign_values: length mismatch");
    }
    L->m->hd_version = h->version;
    return 0;
}

static void hd_free(HdCfg *h)
{
    for (int i = 0; i < h->nnames; i++)
        free(h->names[i]);
    for (int i = 0; i < h->ninv; i++)
        free(h->inv[i]);
    free(h->names);
    free(h->inv);
    free(h->used);
}

static int ld_fail(Ld *L, const char *fmt, const char *a)
{
    snprintf(L->err, L->errlen, fmt, a);
    return -1;
}

static int ld_mat(Ld *L, const char *name, int rows, int cols, int kind,
                  int layer, Q35Mat *M)
{
    const GGUFTensor *t = gguf_tensor(L->g, name);
    if (!t)
        return ld_fail(L, "missing tensor %s", name);
    if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 || t->ne[3] != 1)
        return ld_fail(L, "unexpected shape for %s", name);
    if (t->type != GGML_PTQ1_0 && t->type != GGML_Q8_0 && t->type != GGML_F32
        && t->type != GGML_F16)
        return ld_fail(L, "unsupported type for %s", name);
    memset(M, 0, sizeof(*M));
    M->type = (int)t->type;
    M->rows = rows;
    M->cols = cols;
    M->row_bytes = gguf_type_bytes(t->type, cols);
    M->kind = kind;
    M->layer = layer;
    M->scale = 1.0f;
    uint8_t *d = (uint8_t *)keep(L->m, (size_t)t->nbytes);
    if (!d || gguf_read(L->g, t, 0, d, (size_t)t->nbytes))
        return ld_fail(L, "cannot read %s", name);
    M->data = d;
    L->m->bytes_kind[kind] += t->nbytes;
    if (hd_assign(L, name, M))
        return -1;

    char sname[160];
    const char *dot = strstr(name, ".weight");
    if (dot) {
        snprintf(sname, sizeof(sname), "%.*s.scale", (int)(dot - name), name);
        const GGUFTensor *st = gguf_tensor(L->g, sname);
        if (st && st->type == GGML_F32 && st->nbytes == 4) {
            if (gguf_read(L->g, st, 0, &M->scale, 4))
                return ld_fail(L, "cannot read %s", sname);
            M->has_scale = 1;
        } else if (st) {
            return ld_fail(L, "unsupported scale: %s", sname);
        }
    }
    return 0;
}

static int ld_vec(Ld *L, const char *name, int64_t n, const float **out)
{
    const GGUFTensor *t = gguf_tensor(L->g, name);
    if (!t)
        return ld_fail(L, "missing tensor %s", name);
    if (t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] != n)
        return ld_fail(L, "unexpected shape for %s", name);
    float *d = (float *)keep(L->m, (size_t)n * sizeof(float));
    if (!d)
        return ld_fail(L, "out of memory for %s", name);
    if (t->type == GGML_F32) {
        if (gguf_read(L->g, t, 0, d, (size_t)n * 4))
            return ld_fail(L, "cannot read %s", name);
    } else if (t->type == GGML_F16) {
        uint8_t *h = (uint8_t *)malloc((size_t)n * 2);
        if (!h || gguf_read(L->g, t, 0, h, (size_t)n * 2)) {
            free(h);
            return ld_fail(L, "cannot read %s", name);
        }
        for (int64_t i = 0; i < n; i++)
            d[i] = rd_f16(h + 2 * i);
        free(h);
    } else {
        return ld_fail(L, "unsupported type for %s", name);
    }
    *out = d;
    return 0;
}

static int key_u32(Ld *L, const char *arch, const char *k, uint32_t *v, int need)
{
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", arch, k);
    if (gguf_u32(L->g, key, v)) {
        if (need)
            return ld_fail(L, "missing key %s", key);
        return 1;
    }
    return 0;
}

int q35_load(Q35Model *m, const char *path, char *err, size_t errlen)
{
    memset(m, 0, sizeof(*m));
    GGUF g;
    if (gguf_open(&g, path, err, errlen))
        return -1;
    HdCfg hcfg;
    memset(&hcfg, 0, sizeof(hcfg));
    Ld L = {m, &g, err, errlen, &hcfg};
    Q35Config *c = &m->c;
    const char *arch = gguf_str(&g, "general.architecture");
    int rc = -1;
    if (!arch || strcmp(arch, "qwen35")) {
        ld_fail(&L, "architecture %s: expected qwen35", arch ? arch : "?");
        goto done;
    }
    uint32_t v, nblk, nkv_len = 0, interval = 4, nextn = 0;
    if (key_u32(&L, arch, "embedding_length", &v, 1)) goto done;
    c->n_embd = (int)v;
    if (key_u32(&L, arch, "block_count", &nblk, 1)) goto done;
    if (key_u32(&L, arch, "feed_forward_length", &v, 1)) goto done;
    c->n_ff = (int)v;
    if (key_u32(&L, arch, "attention.head_count", &v, 1)) goto done;
    c->n_head = (int)v;
    if (key_u32(&L, arch, "attention.head_count_kv", &v, 1)) goto done;
    c->n_kv = (int)v;
    if (key_u32(&L, arch, "attention.key_length", &nkv_len, 1)) goto done;
    c->head_dim = (int)nkv_len;
    if (key_u32(&L, arch, "rope.dimension_count", &v, 1)) goto done;
    c->n_rot = (int)v;
    if (key_u32(&L, arch, "ssm.conv_kernel", &v, 1)) goto done;
    c->d_conv = (int)v;
    if (key_u32(&L, arch, "ssm.state_size", &v, 1)) goto done;
    c->d_k = (int)v;
    if (key_u32(&L, arch, "ssm.time_step_rank", &v, 1)) goto done;
    c->v_heads = (int)v;
    if (key_u32(&L, arch, "ssm.group_count", &v, 1)) goto done;
    c->k_heads = (int)v;
    if (key_u32(&L, arch, "ssm.inner_size", &v, 1)) goto done;
    c->d_v = c->v_heads ? (int)v / c->v_heads : 0;
    key_u32(&L, arch, "full_attention_interval", &interval, 0);
    key_u32(&L, arch, "nextn_predict_layers", &nextn, 0);
    c->n_nextn = (int)nextn;
    c->n_layer = (int)nblk - (int)nextn;
    {
        char key[128];
        snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", arch);
        if (gguf_f32(&g, key, &c->eps)) {
            ld_fail(&L, "missing key %s", key);
            goto done;
        }
        snprintf(key, sizeof(key), "%s.rope.freq_base", arch);
        c->rope_base = 10000.0f;
        gguf_f32(&g, key, &c->rope_base);
        snprintf(key, sizeof(key), "%s.rope.dimension_sections", arch);
        int32_t sec[4] = {0, 0, 0, 0};
        const int ns = gguf_arr_i32(&g, key, sec, 4);
        if (ns < 1) {
            sec[0] = c->n_rot / 2;
        }
        for (int i = 0; i < 4; i++)
            c->sect[i] = sec[i];
    }
    if (c->n_layer < 1 || c->n_embd % 128 || c->n_ff % 128 || c->d_k != c->d_v
        || c->n_kv < 1 || c->n_head % c->n_kv || c->v_heads % c->k_heads
        || c->n_rot > c->head_dim || c->n_rot % 2 || c->d_conv < 2) {
        ld_fail(&L, "%s", "unsupported configuration");
        goto done;
    }
    {
        const GGUFTensor *te = gguf_tensor(&g, "token_embd.weight");
        if (!te) {
            ld_fail(&L, "%s", "missing token_embd.weight");
            goto done;
        }
        c->vocab = (int)te->ne[1];
    }
    if (hd_load(&L, &hcfg))
        goto done;
    {
        static const char *known[] = {"version", "tied_output", "block_size", "transform", "axis",
                                      "sign_mode", "sign_widths", "sign_values", "weight_names",
                                      "inverse_weight_names", "gdn_v_grouped"};
        for (uint64_t i = 0; i < g.n_kv; i++) {
            const char *k = g.kv[i].key;
            if (strncmp(k, "prism.", 6) && strncmp(k, "general.tensor_extra", 20))
                continue;
            int ok = 0;
            for (size_t j = 0; j < sizeof(known) / sizeof(known[0]); j++) {
                char full[96];
                snprintf(full, sizeof(full), "prism.hadamard.%s", known[j]);
                ok |= !strcmp(k, full);
            }
            if (!ok)
                fprintf(stderr, "warning: unhandled metadata key: %s\n", k);
        }
    }
    if (ld_mat(&L, "token_embd.weight", c->vocab, c->n_embd, Q35_K_EMBD, -1, &m->embd))
        goto done;
    m->bytes_kind[Q35_K_EMBD] = 0;
    if (gguf_tensor(&g, "output.weight")) {
        if (ld_mat(&L, "output.weight", c->vocab, c->n_embd, Q35_K_HEAD, -1, &m->head))
            goto done;
        m->bytes_kind[Q35_K_EMBD] = (uint64_t)c->vocab * m->embd.row_bytes;
    } else {
        m->head = m->embd;
        m->head.kind = Q35_K_HEAD;
        m->bytes_kind[Q35_K_HEAD] = (uint64_t)c->vocab * m->embd.row_bytes;
    }
    if (hcfg.version) {
        int ok = 1;
        if (hcfg.tied) {
            if (gguf_tensor(&g, "output.weight")) {
                ld_fail(&L, "%s", "prism.hadamard.tied_output with output.weight present");
                goto done;
            }
            memset(&m->head.rot, 0, sizeof(m->head.rot));
            m->head.rot.block = hcfg.block;
            m->head.rot.signs = hd_signs(&hcfg, c->n_embd, &ok);
            m->hd_folded++;
        }
        for (int i = 0; i < hcfg.ninv; i++) {
            if (strcmp(hcfg.inv[i], "token_embd.weight")) {
                ld_fail(&L, "prism.hadamard: unsupported inverse table: %s", hcfg.inv[i]);
                goto done;
            }
            m->embd_inv.block = hcfg.block;
            m->embd_inv.signs = hd_signs(&hcfg, c->n_embd, &ok);
        }
        if (!ok || c->n_embd % hcfg.block) {
            ld_fail(&L, "%s", "prism.hadamard: no signs for the embedding width");
            goto done;
        }
    }
    if (ld_vec(&L, "output_norm.weight", c->n_embd, &m->out_norm))
        goto done;
    m->L = (Q35Layer *)calloc((size_t)c->n_layer, sizeof(Q35Layer));
    if (!m->L) {
        ld_fail(&L, "%s", "out of memory");
        goto done;
    }
    {
        int32_t *recr = (int32_t *)calloc(nblk, sizeof(int32_t));
        char key[128];
        snprintf(key, sizeof(key), "%s.attention.recurrent_layers", arch);
        const int nr = recr ? gguf_arr_i32(&g, key, recr, (int)nblk) : -1;
        for (int l = 0; l < c->n_layer; l++)
            m->L[l].recurrent = nr == (int)nblk ? recr[l] != 0
                                               : (l + 1) % (int)interval != 0;
        free(recr);
    }
    const int E = c->n_embd, F = c->n_ff, hd = c->head_dim;
    const int kd = c->k_heads * c->d_k, vd = c->v_heads * c->d_v, cd = conv_dim(c);
    for (int l = 0; l < c->n_layer; l++) {
        Q35Layer *Ly = &m->L[l];
        char n[128];
#define NM(fmt) (snprintf(n, sizeof(n), "blk.%d." fmt, l), n)
        if (ld_vec(&L, NM("attn_norm.weight"), E, &Ly->attn_norm)
            || ld_vec(&L, NM("post_attention_norm.weight"), E, &Ly->post_norm))
            goto done;
        if (Ly->recurrent) {
            if (ld_mat(&L, NM("attn_qkv.weight"), cd, E, Q35_K_GDN, l, &Ly->wqkv)
                || ld_mat(&L, NM("attn_gate.weight"), vd, E, Q35_K_GDN, l, &Ly->wz)
                || ld_mat(&L, NM("ssm_beta.weight"), c->v_heads, E, Q35_K_GDN_F32, l, &Ly->beta)
                || ld_mat(&L, NM("ssm_alpha.weight"), c->v_heads, E, Q35_K_GDN_F32, l, &Ly->alpha)
                || ld_mat(&L, NM("ssm_out.weight"), E, vd, Q35_K_GDN, l, &Ly->ssm_out)
                || ld_vec(&L, NM("ssm_conv1d.weight"), (int64_t)cd * c->d_conv, &Ly->conv)
                || ld_vec(&L, NM("ssm_a"), c->v_heads, &Ly->ssm_a)
                || ld_vec(&L, NM("ssm_dt.bias"), c->v_heads, &Ly->ssm_dt)
                || ld_vec(&L, NM("ssm_norm.weight"), c->d_v, &Ly->ssm_norm))
                goto done;
            (void)kd;
        } else {
            const int qd = 2 * c->n_head * hd, kvd = c->n_kv * hd;
            if (gguf_tensor(&g, NM("attn_q.weight"))) {
                if (ld_mat(&L, NM("attn_q.weight"), qd, E, Q35_K_ATTN, l, &Ly->wq)
                    || ld_mat(&L, NM("attn_k.weight"), kvd, E, Q35_K_ATTN, l, &Ly->wk)
                    || ld_mat(&L, NM("attn_v.weight"), kvd, E, Q35_K_ATTN, l, &Ly->wv))
                    goto done;
            } else {
                Q35Mat f;
                if (ld_mat(&L, NM("attn_qkv.weight"), qd + 2 * kvd, E, Q35_K_ATTN, l, &f))
                    goto done;
                Ly->wq = Ly->wk = Ly->wv = f;
                Ly->wq.rows = qd;
                Ly->wk.rows = Ly->wv.rows = kvd;
                Ly->wk.data = f.data + (size_t)qd * f.row_bytes;
                Ly->wv.data = f.data + (size_t)(qd + kvd) * f.row_bytes;
            }
            if (ld_mat(&L, NM("attn_output.weight"), E, c->n_head * hd, Q35_K_ATTN, l, &Ly->wo)
                || ld_vec(&L, NM("attn_q_norm.weight"), hd, &Ly->q_norm)
                || ld_vec(&L, NM("attn_k_norm.weight"), hd, &Ly->k_norm))
                goto done;
        }
        if (ld_mat(&L, NM("ffn_gate.weight"), F, E, Q35_K_FFN, l, &Ly->gate)
            || ld_mat(&L, NM("ffn_up.weight"), F, E, Q35_K_FFN, l, &Ly->up)
            || ld_mat(&L, NM("ffn_down.weight"), E, F, Q35_K_FFN, l, &Ly->down))
            goto done;
#undef NM
    }
    for (int i = 0; i < hcfg.nnames; i++)
        if (!hcfg.used[i] && strcmp(hcfg.names[i], "token_embd.weight")) {
            ld_fail(&L, "prism.hadamard: folded weight %s is unused", hcfg.names[i]);
            goto done;
        }
    rc = 0;
done:
    hd_free(&hcfg);
    gguf_close(&g);
    if (rc)
        q35_free(m);
    return rc;
}

void q35_free(Q35Model *m)
{
    for (int i = 0; i < m->nblobs; i++)
        free(m->blobs[i]);
    free(m->blobs);
    free(m->L);
    memset(m, 0, sizeof(*m));
}

void q35_print_config(const Q35Model *m, FILE *f)
{
    const Q35Config *c = &m->c;
    int nrec = 0;
    for (int l = 0; l < c->n_layer; l++)
        nrec += m->L[l].recurrent;
    fprintf(f, "qwen35: embd %d, layers %d (%d GDN + %d attention), ffn %d, vocab %d\n",
            c->n_embd, c->n_layer, nrec, c->n_layer - nrec, c->n_ff, c->vocab);
    fprintf(f, "  attention: %d heads, %d KV, head_dim %d, n_rot %d, base %.0f, "
            "sections %d/%d/%d/%d\n", c->n_head, c->n_kv, c->head_dim, c->n_rot,
            (double)c->rope_base, c->sect[0], c->sect[1], c->sect[2], c->sect[3]);
    fprintf(f, "  GDN: %d k-heads x %d, %d v-heads x %d, conv %d; eps %g; MTP %d\n",
            c->k_heads, c->d_k, c->v_heads, c->d_v, c->d_conv, (double)c->eps, c->n_nextn);
    if (m->hd_version)
        fprintf(f, "  prism.hadamard v%d: %d folded weights, block %d, signs %s, "
                "latent embedding %s, ssm_out %s\n", m->hd_version, m->hd_folded,
                m->head.rot.block ? m->head.rot.block : m->embd_inv.block,
                (m->embd_inv.signs || m->head.rot.signs) ? "explicit" : "identity",
                m->embd_inv.block ? "yes" : "no",
                (c->n_layer && m->L[0].recurrent && m->L[0].ssm_out.rot.perm_rep > 1)
                    ? "permuted" : "unpermuted");
    static const char *kn[Q35_K_N] = {"embd", "head", "attention", "GDN",
                                      "GDN f32", "FFN"};
    uint64_t tot = 0;
    for (int k = 0; k < Q35_K_N; k++) {
        fprintf(f, "  %-9s %10.2f MB\n", kn[k], (double)m->bytes_kind[k] / 1e6);
        if (k != Q35_K_EMBD)
            tot += m->bytes_kind[k];
    }
    fprintf(f, "  weights read per token (excl. embedding row): %.2f MB\n",
            (double)tot / 1e6);
}
