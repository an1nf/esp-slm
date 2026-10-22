"""NumPy reference for the Qwen3.5 hybrid forward pass."""
import argparse
import struct

import numpy as np

import q35_gguf as G

F = np.float32
D = np.float64


def ssum(a, axis=-1):
    return np.take(np.cumsum(a, axis=axis, dtype=F), -1, axis=axis)


def exp32(x):
    return np.exp(np.asarray(x, D)).astype(F)


def log32(x):
    return np.log(np.asarray(x, D)).astype(F)


def silu(x):
    return (x / (F(1) + exp32(-x))).astype(F)


def sigm(x):
    return (F(1) / (F(1) + exp32(-x))).astype(F)


def rmsnorm(x, w, eps):
    mean = ssum(x * x) / F(x.shape[-1])
    sc = F(1) / np.sqrt(mean + F(eps))
    return ((x * sc) * w).astype(F)


def l2norm(x, eps):
    sc = F(1) / np.maximum(np.sqrt(ssum(x * x)), F(eps))
    return (x * sc).astype(F)


def fwht(x, B):
    v = np.asarray(x, F).reshape(-1, B).copy()
    ln = 1
    while ln < B:
        w = v.reshape(v.shape[0], B // (2 * ln), 2, ln)
        a = w[:, :, 0, :].copy()
        b = w[:, :, 1, :].copy()
        w[:, :, 0, :] = a + b
        w[:, :, 1, :] = a - b
        v = w.reshape(-1, B)
        ln *= 2
    return (v * (F(1) / np.sqrt(F(B)))).reshape(-1).astype(F)


class Hadamard:

    def __init__(self, kv):
        self.version = int(kv.get("prism.hadamard.version", 0))
        if not self.version:
            return
        self.block = int(kv["prism.hadamard.block_size"])
        self.tied = bool(kv.get("prism.hadamard.tied_output", False))
        self.grouped = bool(kv.get("prism.hadamard.gdn_v_grouped", False))
        self.names = set(kv["prism.hadamard.weight_names"])
        self.inv = set(kv.get("prism.hadamard.inverse_weight_names", []))
        self.signs = {}
        if kv["prism.hadamard.sign_mode"] == "explicit":
            vals = [int(v) for v in kv["prism.hadamard.sign_values"]]
            off = 0
            for w in kv["prism.hadamard.sign_widths"]:
                w = int(w)
                self.signs[w] = np.asarray(vals[off:off + w], F)
                off += w

    def rot(self, name, cols, perm=None):
        if not self.version or name not in self.names:
            return None
        return (self.block, self.signs.get(cols), perm)


def rot_apply(rot, x):
    block, signs, perm = rot
    x = np.asarray(x, F)
    if perm is not None:
        hd, nk, rep = perm
        x = x.reshape(rep, nk, hd).transpose(1, 0, 2).reshape(-1).copy()
    if signs is not None:
        x = (x * signs).astype(F)
    return fwht(x, block)


class Mat:
    def __init__(self, r, name, rows, cols):
        ttype, ne, raw = r.raw(name)
        assert ne[0] == cols and ne[1] == rows, (name, ne, rows, cols)
        self.type = ttype
        if ttype == G.T_PTQ1_0:
            self.w, self.d = G.ptq1_decode(raw, rows, cols)
        elif ttype == G.T_Q8_0:
            self.w, self.d = G.q8_decode(raw, rows, cols)
        elif ttype == G.T_F32:
            self.w = np.frombuffer(raw, F).reshape(rows, cols)
        elif ttype == G.T_F16:
            self.w = np.frombuffer(raw, np.float16).astype(F).reshape(rows, cols)
        else:
            raise ValueError(name)
        self.rows, self.cols = rows, cols
        sname = name.replace(".weight", ".scale")
        self.scale = None
        self.rot = None
        if sname != name and sname in r.tensors:
            self.scale = F(r.floats(sname)[0])

    def view(self, r0, r1):
        m = Mat.__new__(Mat)
        m.__dict__ = dict(self.__dict__)
        m.w = self.w[r0:r1]
        if hasattr(self, "d"):
            m.d = self.d[r0:r1]
        m.rows = r1 - r0
        return m

    def __call__(self, x):
        cols = self.cols
        if self.rot is not None:
            x = rot_apply(self.rot, x)
        if self.type == G.T_PTQ1_0:
            q, dq = G.q8_act(x)
            nb = cols // 128
            si = (self.w.reshape(self.rows, nb, 4, 32).astype(np.int32)
                  * q.reshape(nb, 4, 32).astype(np.int32)).sum(-1)
            sumf = np.zeros(self.rows, F)
            for b in range(nb):
                sumi = np.zeros(self.rows, F)
                for k in range(4):
                    sumi = sumi + dq[4 * b + k] * si[:, b, k].astype(F)
                sumf = sumf + self.d[:, b] * sumi
            y = sumf
        elif self.type == G.T_Q8_0:
            q, dq = G.q8_act(x)
            nb = cols // 32
            si = (self.w.reshape(self.rows, nb, 32).astype(np.int32)
                  * q.reshape(nb, 32).astype(np.int32)).sum(-1)
            y = np.zeros(self.rows, F)
            for b in range(nb):
                y = y + si[:, b].astype(F) * (self.d[:, b] * dq[b])
        elif self.type == G.T_F32:
            y = ssum(self.w * x[None, :].astype(F), axis=1)
        else:
            y = ssum(self.w * x[None, :].astype(F), axis=1)
        y = y.astype(F)
        if self.scale is not None:
            y = (y * self.scale).astype(F)
        return y


def embed_row(m, tok):
    if m.type == G.T_PTQ1_0:
        return (m.w[tok].astype(F).reshape(-1, 128) * m.d[tok][:, None]).reshape(-1).astype(F)
    if m.type == G.T_Q8_0:
        return (m.w[tok].astype(F).reshape(-1, 32) * m.d[tok][:, None]).reshape(-1).astype(F)
    return m.w[tok].astype(F)


class Model:
    def __init__(self, path):
        r = G.Reader(path)
        kv = r.kv
        a = kv["general.architecture"]
        assert a == "qwen35"
        g = lambda k, d=None: kv.get(f"{a}.{k}", d)
        nextn = int(g("nextn_predict_layers", 0))
        self.n_layer = int(g("block_count")) - nextn
        self.E = int(g("embedding_length"))
        self.Fd = int(g("feed_forward_length"))
        self.nh = int(g("attention.head_count"))
        self.nkv = int(g("attention.head_count_kv"))
        self.hd = int(g("attention.key_length"))
        self.n_rot = int(g("rope.dimension_count"))
        self.base = F(g("rope.freq_base", 10000.0))
        self.eps = F(g("attention.layer_norm_rms_epsilon"))
        self.K = int(g("ssm.conv_kernel"))
        self.dk = int(g("ssm.state_size"))
        self.Hv = int(g("ssm.time_step_rank"))
        self.Hk = int(g("ssm.group_count"))
        self.dv = int(g("ssm.inner_size")) // self.Hv
        interval = int(g("full_attention_interval", 4))
        sec = g("rope.dimension_sections")
        self.sect = [int(s) for s in sec] + [0] * (4 - len(sec)) if sec is not None else [self.n_rot // 2, 0, 0, 0]
        te = r.tensors["token_embd.weight"]
        self.V = int(te[1][1])
        E, Fd, hd = self.E, self.Fd, self.hd
        self.hdm = H = Hadamard(kv)
        self.embd = Mat(r, "token_embd.weight", self.V, E)
        if "output.weight" in r.tensors:
            self.head = Mat(r, "output.weight", self.V, E)
            self.head.rot = H.rot("output.weight", E)
        else:
            self.head = Mat.__new__(Mat)
            self.head.__dict__ = dict(self.embd.__dict__)
            self.head.rot = (H.block, H.signs.get(E), None) if H.version and H.tied else None
        self.embd_inv = (H.block, H.signs.get(E), None) if H.version and "token_embd.weight" in H.inv else None
        def M(name, rows, cols, perm=None):
            m = Mat(r, name, rows, cols)
            m.rot = H.rot(name, cols, perm)
            return m
        self.out_norm = r.floats("output_norm.weight")
        kd, vd = self.Hk * self.dk, self.Hv * self.dv
        cd = 2 * kd + vd
        self.L = []
        for l in range(self.n_layer):
            n = lambda s: f"blk.{l}.{s}"
            L = {"rec": (l + 1) % interval != 0,
                 "attn_norm": r.floats(n("attn_norm.weight")),
                 "post_norm": r.floats(n("post_attention_norm.weight"))}
            if L["rec"]:
                L["wqkv"] = M(n("attn_qkv.weight"), cd, E)
                L["wz"] = M(n("attn_gate.weight"), vd, E)
                L["beta"] = Mat(r, n("ssm_beta.weight"), self.Hv, E)
                L["alpha"] = Mat(r, n("ssm_alpha.weight"), self.Hv, E)
                perm = (self.dv, self.Hk, self.Hv // self.Hk) if H.version and H.grouped and self.Hv > self.Hk else None
                L["out"] = M(n("ssm_out.weight"), E, vd, perm)
                L["conv"] = r.floats(n("ssm_conv1d.weight")).reshape(cd, self.K)
                L["a"] = r.floats(n("ssm_a"))
                L["dt"] = r.floats(n("ssm_dt.bias"))
                L["norm"] = r.floats(n("ssm_norm.weight"))
            else:
                qd, kvd = 2 * self.nh * hd, self.nkv * hd
                if n("attn_q.weight") in r.tensors:
                    L["wq"] = M(n("attn_q.weight"), qd, E)
                    L["wk"] = M(n("attn_k.weight"), kvd, E)
                    L["wv"] = M(n("attn_v.weight"), kvd, E)
                else:
                    f = M(n("attn_qkv.weight"), qd + 2 * kvd, E)
                    L["wq"], L["wk"], L["wv"] = f.view(0, qd), f.view(qd, qd + kvd), f.view(qd + kvd, qd + 2 * kvd)
                L["wo"] = M(n("attn_output.weight"), E, self.nh * hd)
                L["qn"] = r.floats(n("attn_q_norm.weight"))
                L["kn"] = r.floats(n("attn_k_norm.weight"))
            L["gate"] = M(n("ffn_gate.weight"), Fd, E)
            L["up"] = M(n("ffn_up.weight"), Fd, E)
            L["down"] = M(n("ffn_down.weight"), E, Fd)
            self.L.append(L)

    def rope_cache(self, pos):
        half = self.n_rot // 2
        ts = F(np.power(D(self.base), D(F(-2.0) / F(self.n_rot))))
        s = self.sect
        sd = sum(s)
        th, te = F(pos), F(0)
        c, sn = np.zeros(half, F), np.zeros(half, F)
        for i in range(half):
            sec = i % sd
            is_e = not ((sec % 3 == 1 and sec < 3 * s[1]) or (sec % 3 == 2 and sec < 3 * s[2])
                        or (sec % 3 == 0 and sec < 3 * s[0]))
            a = te if is_e else th
            c[i] = F(np.cos(D(a)))
            sn[i] = F(np.sin(D(a)))
            th = F(th * ts)
            te = F(te * ts)
        return c, sn

    @staticmethod
    def rope(v, c, s):
        half = len(c)
        x0, x1 = v[:half].copy(), v[half:2 * half].copy()
        v[:half] = x0 * c - x1 * s
        v[half:2 * half] = x0 * s + x1 * c

    def run(self, ids, kv_f16=True, dbg=None):
        put = (lambda k, l, p, v: dbg.__setitem__((k, l, p), np.array(v, copy=True))) if dbg is not None \
            else (lambda *a: None)
        hd, nh, nkv = self.hd, self.nh, self.nkv
        kd, vd = self.Hk * self.dk, self.Hv * self.dv
        cd = 2 * kd + vd
        conv = {l: np.zeros((self.K - 1, cd), F) for l in range(self.n_layer) if self.L[l]["rec"]}
        S = {l: np.zeros((self.Hv, self.dv, self.dk), F) for l in conv}
        Kc = {l: [] for l in range(self.n_layer) if not self.L[l]["rec"]}
        Vc = {l: [] for l in Kc}
        out = []
        scale_v = F(1) / np.sqrt(F(self.dv))
        scale_a = F(1) / np.sqrt(F(hd))
        for pos, tok in enumerate(ids):
            x = embed_row(self.embd, tok)
            if self.embd_inv is not None:
                block, signs, _ = self.embd_inv
                x = fwht(x, block)
                if signs is not None:
                    x = (x * signs).astype(F)
            for l, L in enumerate(self.L):
                xn = rmsnorm(x, L["attn_norm"], self.eps)
                if L["rec"]:
                    qkv = L["wqkv"](xn)
                    z = L["wz"](xn)
                    bt = L["beta"](xn)
                    al = L["alpha"](xn)
                    cs, w = conv[l], L["conv"]
                    acc = np.zeros(cd, F)
                    for k in range(self.K - 1):
                        acc = acc + cs[k] * w[:, k]
                    acc = acc + qkv * w[:, self.K - 1]
                    co = silu(acc)
                    put("linear_attn_qkv_mixed", l, pos, qkv)
                    put("conv_output_silu", l, pos, co)
                    put("z", l, pos, z)
                    conv[l] = np.concatenate([cs[1:], qkv[None, :]], 0)
                    qk = co[:2 * kd].reshape(2 * self.Hk, self.dk)
                    qk = np.stack([l2norm(r_, self.eps) for r_ in qk])
                    put("qk_conv_l2", l, pos, qk)
                    qv, kv_ = qk[:self.Hk], qk[self.Hk:]
                    vv = co[2 * kd:].reshape(self.Hv, self.dv)
                    o = np.zeros((self.Hv, self.dv), F)
                    for h in range(self.Hv):
                        q, k, v = qv[h % self.Hk], kv_[h % self.Hk], vv[h]
                        beta = sigm(bt[h])
                        xa = F(al[h] + L["dt"][h])
                        sp = xa if xa > F(20) else log32(F(1) + exp32(xa))
                        eg = exp32(F(L["a"][h] * sp))
                        put("beta", l, (pos, h), beta)
                        put("gate", l, (pos, h), F(L["a"][h] * sp))
                        M = (S[l][h] * eg).astype(F)
                        sums = ssum(M * k[None, :], axis=1)
                        delta = ((v - sums) * beta).astype(F)
                        M = (M + k[None, :] * delta[:, None]).astype(F)
                        S[l][h] = M
                        oh = (ssum(M * q[None, :], axis=1) * scale_v).astype(F)
                        put("attn_output", l, (pos, h), oh)
                        oh = rmsnorm(oh, L["norm"], self.eps)
                        o[h] = (silu(z[h * self.dv:(h + 1) * self.dv]) * oh).astype(F)
                    put("final_output", l, pos, o.reshape(-1))
                    y = L["out"](o.reshape(-1))
                    put("linear_attn_out", l, pos, y)
                else:
                    qg = L["wq"](xn).reshape(nh, 2 * hd)
                    k = L["wk"](xn).reshape(nkv, hd)
                    v = L["wv"](xn).reshape(nkv, hd)
                    c, s = self.rope_cache(pos)
                    q = np.zeros((nh, hd), F)
                    for h in range(nh):
                        q[h] = rmsnorm(qg[h, :hd], L["qn"], self.eps)
                        self.rope(q[h], c, s)
                    for h in range(nkv):
                        k[h] = rmsnorm(k[h], L["kn"], self.eps)
                        self.rope(k[h], c, s)
                    if kv_f16:
                        k = k.astype(np.float16).astype(F)
                        v = v.astype(np.float16).astype(F)
                    Kc[l].append(k)
                    Vc[l].append(v)
                    Ks, Vs = np.stack(Kc[l]), np.stack(Vc[l])
                    att = np.zeros((nh, hd), F)
                    grp = nh // nkv
                    for h in range(nh):
                        kh = h // grp
                        sc = (ssum(Ks[:, kh] * q[h][None, :], axis=1) * scale_a).astype(F)
                        e = exp32(sc - sc.max())
                        pr = (e * (F(1) / ssum(e))).astype(F)
                        oh = ssum(pr[:, None] * Vs[:, kh], axis=0)
                        att[h] = (oh * sigm(qg[h, hd:])).astype(F)
                    y = L["wo"](att.reshape(-1))
                x = (x + y).astype(F)
                put("attn_residual", l, pos, x)
                xn = rmsnorm(x, L["post_norm"], self.eps)
                put("attn_post_norm", l, pos, xn)
                g = L["gate"](xn)
                u = L["up"](xn)
                put("ffn_swiglu", l, pos, (silu(g) * u).astype(F))
                y = L["down"]((silu(g) * u).astype(F))
                put("ffn_out", l, pos, y)
                x = (x + y).astype(F)
                put("l_out", l, pos, x)
            xn = rmsnorm(x, self.out_norm, self.eps)
            out.append(self.head(xn))
        return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--kv", default="f16")
    a = ap.parse_args()
    ids = [int(t) for t in a.ids.split(",")]
    m = Model(a.model)
    kv16 = a.kv != "f32"
    lg = m.run(ids, kv16)
    with open(a.out, "wb") as f:
        f.write(b"Q35R" + struct.pack("<III", len(ids), m.V, int(kv16)))
        f.write(np.asarray(ids, np.int32).tobytes())
        f.write(lg.astype(F).tobytes())


if __name__ == "__main__":
    main()
