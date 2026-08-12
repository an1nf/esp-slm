"""NumPy reference for BitNet b1.58 (.tbin)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tbin import HEAD_F32, HEAD_I8, quant_rows_i8

F = np.float32
KV_F32, KV_I8 = 0, 1


def ssum(a, axis=-1):
    return np.take(np.cumsum(a, axis=axis, dtype=F), -1, axis=axis)


def rmsnorm(x, w, eps):
    ss = ssum(x * x) / F(x.shape[-1])
    r = F(1.0) / np.sqrt(ss + F(eps))
    return (w * (x * r)).astype(F)


def quant_act(x):
    m = np.max(np.abs(x))
    s = F(127.0) / max(m, F(1e-5))
    xq = np.clip(np.rint(x * s), -128, 127).astype(np.int32)
    return xq, F(s)


def bitlinear(w_i32, scale, xq, s):
    acc = w_i32 @ xq
    return ((acc.astype(F) / s) * F(scale)).astype(F)


def kv_quant(v, nkv, hd):
    v = v.reshape(nkv, hd)
    m = np.max(np.abs(v), axis=1)
    sc = np.where(m > 0, m / F(127.0), F(1.0)).astype(F)
    q = np.clip(np.rint(v / sc[:, None]), -127, 127).astype(np.int8)
    return q.reshape(-1), sc


def rope(v, n_heads, hd, pos, theta):
    v = v.reshape(n_heads, hd).copy()
    half = hd // 2
    i = np.arange(half, dtype=F)
    inv = F(1.0) / np.power(F(theta), (F(2.0) * i) / F(hd))
    ang = F(pos) * inv
    c, s = np.cos(ang).astype(F), np.sin(ang).astype(F)
    x1, x2 = v[:, :half].copy(), v[:, half:].copy()
    v[:, :half] = x1 * c - x2 * s
    v[:, half:] = x2 * c + x1 * s
    return v.reshape(-1)


class RefModel:
    def __init__(self, cfg: dict, params: dict, head_fmt: int = HEAD_F32,
                 kv_fmt: int = KV_F32):
        self.c = cfg
        self.p = params
        self.head_fmt, self.kv_fmt = head_fmt, kv_fmt
        if head_fmt == HEAD_I8:
            q, sc = quant_rows_i8(params["tok_emb"])
            self.hq, self.hsc = q.astype(np.int32), sc
            self.emb = (q.astype(F) * sc[:, None]).astype(F)
        else:
            self.emb = params["tok_emb"].astype(F)
        self.W = [{n: lay[n][0].astype(np.int32) for n in
                   ("q", "k", "v", "o", "gate", "up", "down")}
                  for lay in params["layers"]]
        kvdim = cfg["n_kv"] * cfg["head_dim"]
        self.kc = np.zeros((cfg["n_layers"], cfg["max_seq"], kvdim), F)
        self.vc = np.zeros_like(self.kc)

    def forward(self, token: int, pos: int) -> np.ndarray:
        c, p = self.c, self.p
        nh, nkv, hd = c["n_heads"], c["n_kv"], c["head_dim"]
        group, scale = nh // nkv, F(1.0 / np.sqrt(hd))
        x = self.emb[token].copy()
        for l, lay in enumerate(p["layers"]):
            W = self.W[l]
            xn = rmsnorm(x, lay["attn_norm"], c["eps"])
            xq, s = quant_act(xn)
            q = bitlinear(W["q"], lay["q"][1], xq, s)
            k = bitlinear(W["k"], lay["k"][1], xq, s)
            v = bitlinear(W["v"], lay["v"][1], xq, s)
            q = rope(q, nh, hd, pos, c["rope_theta"])
            k = rope(k, nkv, hd, pos, c["rope_theta"])
            if self.kv_fmt == KV_I8:
                kq, ks = kv_quant(k, nkv, hd)
                vq, vs = kv_quant(v, nkv, hd)
                k = (kq.reshape(nkv, hd).astype(F) * ks[:, None]).reshape(-1)
                v = (vq.reshape(nkv, hd).astype(F) * vs[:, None]).reshape(-1)
            self.kc[l, pos], self.vc[l, pos] = k, v
            att = np.zeros(nh * hd, F)
            for h in range(nh):
                kh = h // group
                K = self.kc[l, :pos + 1, kh * hd:(kh + 1) * hd]
                V = self.vc[l, :pos + 1, kh * hd:(kh + 1) * hd]
                sc = ssum(K * q[None, h * hd:(h + 1) * hd], axis=1) * scale
                e = np.exp(sc - sc.max()).astype(F)
                pr = (e / ssum(e)).astype(F)
                att[h * hd:(h + 1) * hd] = ssum(pr[:, None] * V, axis=0)
            att = rmsnorm(att, lay["attn_sub_norm"], c["eps"])
            xq, s = quant_act(att)
            x = (x + bitlinear(W["o"], lay["o"][1], xq, s)).astype(F)
            xn = rmsnorm(x, lay["ffn_norm"], c["eps"])
            xq, s = quant_act(xn)
            g = bitlinear(W["gate"], lay["gate"][1], xq, s)
            u = bitlinear(W["up"], lay["up"][1], xq, s)
            r = np.maximum(g, F(0.0))
            hh = rmsnorm((r * r) * u, lay["ffn_sub_norm"], c["eps"])
            xq, s = quant_act(hh)
            x = (x + bitlinear(W["down"], lay["down"][1], xq, s)).astype(F)
        xn = rmsnorm(x, p["final_norm"], c["eps"])
        if self.head_fmt == HEAD_I8:
            xq, s = quant_act(xn)
            acc = self.hq @ xq
            return ((acc.astype(F) / s) * self.hsc).astype(F)
        return ssum(self.emb * xn[None, :], axis=1)


def write_ref(path: str, vocab: int, prompt, gen, logits) -> None:
    import struct
    prompt = np.asarray(prompt, np.int32)
    gen = np.asarray(gen, np.int32)
    logits = np.asarray(logits, np.float32)
    assert logits.shape == (len(prompt) + len(gen) - 1, vocab)
    with open(path, "wb") as f:
        f.write(b"TREF" + struct.pack("<3I", vocab, len(prompt), len(gen)))
        f.write(prompt.tobytes())
        f.write(gen.tobytes())
        f.write(logits.tobytes())


def greedy_reference(model: RefModel, prompt, n_gen: int):
    logits, gen = [], []
    pos = 0
    for t in prompt:
        lg = model.forward(int(t), pos)
        logits.append(lg)
        pos += 1
    nxt = int(np.argmax(logits[-1]))
    for i in range(n_gen):
        gen.append(nxt)
        if i == n_gen - 1:
            break
        lg = model.forward(nxt, pos)
        logits.append(lg)
        pos += 1
        nxt = int(np.argmax(lg))
    return gen, np.stack(logits)
