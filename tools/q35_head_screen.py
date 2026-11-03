"""Head preselection by low-rank or block previews."""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_cmp as C
import q35_gguf as G
from q35_head_bench import EXTRA
from q35_spec_bench import PROMPTS

MS = (16, 64, 256, 1024, 4096)
RS = (32, 64, 128, 256)
BS = (1, 2, 4)
CHUNK = 8192


def head_rows(path):
    r = G.Reader(path)
    name = "output.weight" if "output.weight" in r.tensors else "token_embd.weight"
    info = r.tensors[name]
    cols, rows = int(info[1][0]), int(info[1][1])
    if int(info[0]) != 143:
        sys.exit(f"{name}: tipo {info[0]}, se esperaba PTQ1_0 (143)")
    raw = r.raw(name)[2]
    return raw, rows, cols


def decode_chunk(raw, r0, r1, cols):
    rb = cols // 128 * 28
    t, d = G.ptq1_decode(raw[r0 * rb:r1 * rb], r1 - r0, cols)
    return t, d


def topk_merge(vals, ids, new_vals, new_ids, k):
    v = np.concatenate([vals, new_vals], 0)
    i = np.concatenate([ids, new_ids], 0)
    if v.shape[0] <= k:
        return v, i
    sel = np.argpartition(-v, k - 1, axis=0)[:k]
    return np.take_along_axis(v, sel, 0), np.take_along_axis(i, sel, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--q35", default="build/q35")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0]
    raw, V, E = head_rows(a.model)
    nb = E // 128
    row_bytes = nb * 28
    print(f"head {V} x {E} ({V * row_bytes / 1e6:.1f} MB)", flush=True)

    texts = dict(PROMPTS)
    texts.update(EXTRA)
    HX, want, ctxs = [], [], []
    for name, text in texts.items():
        if a.llama == "none":
            ids = [(7 * k + len(name)) % 97 + 1 for k in range(12)]
        else:
            tok = os.path.join(a.out, f"{tag}.{name}.ids.bin")
            r = subprocess.run([a.llama, a.model, tok, "--text", text, "--tokens-only", "1"],
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stderr[-2000:])
            ids, _ = C.load(tok)
            ids = [int(t) for t in ids]
        print(f"{name}: generating", flush=True)
        r = subprocess.run([a.q35, "spec", a.model, "--ids", ",".join(map(str, ids)),
                            "--gen", str(a.gen), "--ctx", str(len(ids) + a.gen + 40),
                            "--drafter", "mix", "--gamma", "8", "--stop"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout + r.stderr[-2000:])
        gen = [int(t) for t in re.search(r"^ids: (.*)$", r.stdout, re.M).group(1).split(",")]
        seq = ids + gen
        hd = os.path.join(a.out, f"{tag}.{name}.hx.bin")
        env = dict(os.environ, Q35_HDUMP=hd)
        r = subprocess.run([a.q35, "dump", a.model, os.devnull, "--ids", ",".join(map(str, seq)),
                            "--ctx", str(len(seq) + 8), "--chunk", "16"],
                           capture_output=True, text=True, env=env)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        hx = np.fromfile(hd, np.float32).reshape(len(seq), E)
        os.remove(hd)
        n0 = len(ids)
        for p in range(n0 - 1, len(seq) - 1):
            HX.append(hx[p])
            want.append(seq[p + 1])
            ctxs.append(np.unique(np.asarray(seq[:p + 1])))
    HX = np.stack(HX).astype(np.float32)
    P = HX.shape[0]
    want = np.asarray(want)
    print(f"{P} posiciones", flush=True)

    G2 = np.zeros((E, E), np.float64)
    for r0 in range(0, V, CHUNK):
        r1 = min(V, r0 + CHUNK)
        t, d = decode_chunk(raw, r0, r1, E)
        W = (t.reshape(r1 - r0, nb, 128) * d[:, :, None]).reshape(r1 - r0, E).astype(np.float32)
        G2 += (W.T @ W).astype(np.float64)
    ev, U = np.linalg.eigh(G2)
    U = U[:, ::-1].astype(np.float32)
    ev = ev[::-1]
    energy = np.cumsum(ev) / ev.sum()
    print("energy of W^T W in top r: " + ", ".join(f"r={r}: {energy[r - 1]:.3f}" for r in RS),
          flush=True)
    Z = {r: HX @ U[:, :r] for r in RS}

    KX = 1024
    MM = max(MS)
    ex_v = np.full((0, P), -np.inf, np.float32)
    ex_i = np.zeros((0, P), np.int64)
    lse = np.full(P, -np.inf)
    bs = [b for b in BS if b < nb]
    meth = [f"svd {r}" for r in RS] + [f"blk {b}" for b in bs]
    pv = {k: (np.full((0, P), -np.inf, np.float32), np.zeros((0, P), np.int64)) for k in meth}
    for r0 in range(0, V, CHUNK):
        r1 = min(V, r0 + CHUNK)
        t, d = decode_chunk(raw, r0, r1, E)
        n = r1 - r0
        W = (t.reshape(n, nb, 128) * d[:, :, None]).reshape(n, E).astype(np.float32)
        L = W @ HX.T
        ids = np.arange(r0, r1)[:, None].repeat(P, 1)
        ex_v, ex_i = topk_merge(ex_v, ex_i, L, ids, KX)
        m = L.max(0)
        lse = np.logaddexp(lse, m + np.log(np.exp(L - m).sum(0)))
        for r in RS:
            Pr = W @ U[:, :r]
            sc = np.abs(Pr).max(1, keepdims=True) / 127.0
            sc[sc == 0] = 1.0
            Pq = np.round(Pr / sc) * sc
            pv[f"svd {r}"] = topk_merge(*pv[f"svd {r}"], Pq @ Z[r].T, ids, MM)
        for b in bs:
            Wb = (t[:, :b * 128].reshape(n, b, 128) * d[:, :b, None]).reshape(n, b * 128)
            pv[f"blk {b}"] = topk_merge(*pv[f"blk {b}"], Wb.astype(np.float32) @ HX[:, :b * 128].T, ids, MM)
        if r0 // CHUNK % 8 == 0:
            print(f"  rows {r1}/{V}", flush=True)
    order = np.argsort(-ex_v, 0)
    ex_i = np.take_along_axis(ex_i, order, 0)
    ex_v = np.take_along_axis(ex_v, order, 0)
    amax = ex_i[0]
    agree = float((amax == want).mean())
    top20 = ex_i[:20]
    pex = np.exp(ex_v.astype(np.float64) - lse[None, :])
    rows = []
    for k in meth:
        v, i = pv[k]
        o = np.argsort(-v, 0)
        i = np.take_along_axis(i, o, 0)
        for M in MS:
            cand = i[:M]
            hit1, hit20, hit1c, mass = [], [], [], []
            sets_ = []
            for p in range(P):
                s = set(cand[:, p].tolist())
                hit1.append(amax[p] in s)
                hit20.append(all(int(x) in s for x in top20[:, p]))
                sc = s | set(ctxs[p].tolist())
                hit1c.append(amax[p] in sc)
                sets_.append(np.fromiter(sc, np.int64))
            if M in (64, 256):
                mass = [float(pex[np.isin(ex_i[:, p], sets_[p]), p].sum()) for p in range(P)]
            uni = {}
            for N in (1, 8, 16):
                us = []
                for p0 in range(0, P - N + 1, N):
                    u = set()
                    for p in range(p0, p0 + N):
                        u |= set(cand[:, p].tolist())
                    us.append(len(u))
                uni[N] = float(np.mean(us)) if us else float("nan")
            prev = (V * int(k.split()[1]) + 2 * V) if k.startswith("svd") else V * row_bytes * int(k.split()[1]) / nb
            rows.append(dict(k=k, M=M, h1=np.mean(hit1), h20=np.mean(hit20), h1c=np.mean(hit1c),
                             mass=(np.mean(mass) if mass else None), prev=prev / (V * row_bytes),
                             u1=uni[1] * row_bytes / (V * row_bytes), u16=uni[16] * row_bytes / (V * row_bytes)))
            print(f"  {k} M={M}: argmax {rows[-1]['h1']:.4f}, top-20 {rows[-1]['h20']:.4f}", flush=True)
    L = [f"# Head preselection, {tag}", "",
         f"{P} generated positions (greedy, {len(texts)} prompts, up to {a.gen} tokens). Head "
         f"{V} x {E}, {V * row_bytes / 1e6:.1f} MB. Exact float logits; argmax matches the q35 token "
         f"at {100 * agree:.2f} % of positions (the rest are near ties decided by Q8 rounding).", "",
         "Energy of W^T W in the top r directions: "
         + ", ".join(f"r = {r}: {100 * energy[r - 1]:.1f} %" for r in RS) + ".", "",
         "Bytes read per pass as a fraction of the full head: preview + candidate rows "
         "(1 lane, or union over 16 consecutive positions).", "",
         "| Preview | M | argmax in cands | +ctx | full top-20 | mass covered (+ctx) | "
         "preview | + M rows (1 lane) | + union (16 lanes) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        ms = "" if x["mass"] is None else f"{x['mass']:.4f}"
        L.append(f"| {x['k']} | {x['M']} | {100 * x['h1']:.2f} % | {100 * x['h1c']:.2f} % | "
                 f"{100 * x['h20']:.2f} % | {ms} | {x['prev']:.3f} | {x['prev'] + x['u1']:.3f} | "
                 f"{x['prev'] + x['u16']:.3f} |")
    rep = "\n".join(L)
    with open(os.path.join(a.out, f"{tag}.screen.md"), "w") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
