"""Structure of the head input after Hadamard folding."""
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
from q35_head_screen import decode_chunk, head_rows, topk_merge
from q35_spec_bench import PROMPTS

MS = (64, 256, 1024)
CHUNK = 2048


def gather(a, tag, E):
    cache = os.path.join(a.out, f"{tag}.hx.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["HX"], z["want"], list(z["ctxs"])
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
            ids = [int(t) for t in C.load(tok)[0]]
        print(f"{name}: generating", flush=True)
        r = subprocess.run([a.q35, "spec", a.model, "--ids", ",".join(map(str, ids)),
                            "--gen", str(a.gen), "--ctx", str(len(ids) + a.gen + 40),
                            "--drafter", "mix", "--gamma", "8", "--stop"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout + r.stderr[-2000:])
        gen = [int(t) for t in re.search(r"^ids: (.*)$", r.stdout, re.M).group(1).split(",")]
        seq = ids + gen
        hd = os.path.join(a.out, f"{tag}.{name}.hx.bin")
        r = subprocess.run([a.q35, "dump", a.model, os.devnull, "--ids", ",".join(map(str, seq)),
                            "--ctx", str(len(seq) + 8), "--chunk", "16"],
                           capture_output=True, text=True, env=dict(os.environ, Q35_HDUMP=hd))
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        hx = np.fromfile(hd, np.float32).reshape(len(seq), E)
        os.remove(hd)
        for p in range(len(ids) - 1, len(seq) - 1):
            HX.append(hx[p])
            want.append(seq[p + 1])
            ctxs.append(np.unique(np.asarray(seq[:p + 1])))
    HX = np.stack(HX).astype(np.float32)
    want = np.asarray(want)
    np.savez(cache, HX=HX, want=want, ctxs=np.array(ctxs, dtype=object))
    return HX, want, ctxs


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
    rd = G.Reader(a.model)
    gb = int(rd.kv.get("prism.hadamard.block_size", E) or E)
    if gb <= 0 or E % gb:
        gb = E
    NG = E // gb
    HX, want, ctxs = gather(a, tag, E)
    P = HX.shape[0]
    print(f"head {V} x {E}; Hadamard groups of {gb} ({NG} groups); {P} positions", flush=True)

    eg = (HX.reshape(P, NG, gb) ** 2).sum(2)
    fg = eg / eg.sum(1, keepdims=True)
    eb = (HX.reshape(P, nb, 128) ** 2).sum(2)
    top_g = fg.argmax(1)
    s1 = dict(mean=fg.mean(0), maxfrac=float(fg.max(1).mean()), dom=float((fg.max(1) > 2.0 / NG).mean()),
              stable=float((top_g == np.bincount(top_g, minlength=NG).argmax()).mean()))
    cv = []
    if gb >= 256:
        bpg = gb // 128
        ebg = eb.reshape(P, NG, bpg)
        cv = (ebg.std(2) / ebg.mean(2)).mean()
    methods = {}
    bpg = max(1, gb // 128)
    for b in (1, 2, 4):
        if b >= nb:
            continue
        m = np.zeros((nb, P), bool)
        m[:b] = True
        methods[f"fixed {b}"] = m
        m = np.zeros((nb, P), bool)
        step = max(1, nb // b)
        m[[min(nb - 1, k * step) for k in range(b)]] = True
        methods[f"strided {b}"] = m
        m = np.zeros((nb, P), bool)
        top = np.argsort(-eb, 1)[:, :b]
        m[top.T, np.arange(P)[None, :]] = True
        methods[f"energy {b}"] = m
        m = np.zeros((nb, P), bool)
        for p0 in range(0, P, 16):
            w = eb[p0:p0 + 16].sum(0)
            m[np.argsort(-w)[:b], p0:p0 + 16] = True
        methods[f"energy16 {b}"] = m
    hxb = np.sqrt(eb).T
    MM = max(MS)
    pv = {k: (np.full((0, P), -np.inf, np.float32), np.zeros((0, P), np.int64)) for k in methods}
    ex_v = np.full((0, P), -np.inf, np.float32)
    ex_i = np.zeros((0, P), np.int64)
    lse = np.full(P, -np.inf)
    rowfrac = []
    HXb = HX.reshape(P, nb, 128)
    for r0 in range(0, V, CHUNK):
        r1 = min(V, r0 + CHUNK)
        n = r1 - r0
        t, d = decode_chunk(raw, r0, r1, E)
        Wb = t.reshape(n, nb, 128).astype(np.float32) * d[:, :, None]
        part = np.einsum("nbk,pbk->bnp", Wb, HXb, optimize=True)
        L = part.sum(0)
        ids = np.arange(r0, r1)[:, None].repeat(P, 1)
        ex_v, ex_i = topk_merge(ex_v, ex_i, L, ids, 1024)
        mx = L.max(0)
        lse = np.logaddexp(lse, mx + np.log(np.exp(L - mx).sum(0)))
        for k, m in methods.items():
            sc = np.einsum("bnp,bp->np", part, m.astype(np.float32), optimize=True)
            pv[k] = topk_merge(*pv[k], sc, ids, MM)
        wg = (Wb.reshape(n, NG, -1) ** 2).sum(2)
        rowfrac.append((wg / np.maximum(wg.sum(1, keepdims=True), 1e-30)).max(1))
        if r0 // CHUNK % 32 == 0:
            print(f"  rows {r1}/{V}", flush=True)
    o = np.argsort(-ex_v, 0)
    ex_i = np.take_along_axis(ex_i, o, 0)
    ex_v = np.take_along_axis(ex_v, o, 0)
    amax, lmax = ex_i[0], ex_v[0]
    top20 = ex_i[:20]
    rowfrac = np.concatenate(rowfrac)
    res = []
    cand_sets = {}
    for k in methods:
        v, i = pv[k]
        i = np.take_along_axis(i, np.argsort(-v, 0), 0)
        for M in MS:
            h1c, h20, loss = [], [], []
            for p in range(P):
                s = set(i[:M, p].tolist()) | set(ctxs[p].tolist())
                h1c.append(amax[p] in s)
                h20.append(all(int(x) in s for x in top20[:, p]))
                if amax[p] in s:
                    loss.append(0.0)
                else:
                    inS = np.isin(ex_i[:, p], np.fromiter(s, np.int64))
                    best = ex_v[inS, p].max() if inS.any() else ex_v[-1, p]
                    loss.append(float(lmax[p] - best))
            res.append((k, M, np.mean(h1c), np.mean(h20), np.mean(loss)))
            if M == 1024:
                cand_sets[k] = i[:M]
    cert = {}
    for k in [x for x in methods if x.startswith(("fixed 4", "energy 4"))]:
        m = methods[k]
        unc = np.zeros(P)
        for r0 in range(0, V, CHUNK):
            r1 = min(V, r0 + CHUNK)
            n = r1 - r0
            t, d = decode_chunk(raw, r0, r1, E)
            Wb = t.reshape(n, nb, 128).astype(np.float32) * d[:, :, None]
            part = np.einsum("nbk,pbk->bnp", Wb, HXb, optimize=True)
            sc = np.einsum("bnp,bp->np", part, m.astype(np.float32), optimize=True)
            wn = np.sqrt((Wb ** 2).sum(2))
            R = wn @ (hxb * (~m))
            unc += ((sc + R) >= lmax[None, :]).sum(0)
        cert[k] = (float(np.mean(unc / V)), float(np.mean(unc <= 1024)))
    L_ = [f"# Head input structure, {tag}", "",
          f"{P} generated positions (greedy, {len(dict(PROMPTS, **EXTRA))} prompts). Head {V} x {E}; "
          f"Hadamard groups of {gb} columns ({NG} groups, {bpg} blocks of 128 per group).", "",
          "## Energy split", "",
          "| | value | if uniform |", "|---|---|---|",
          "| mean fraction per group | " + ", ".join(f"{x:.3f}" for x in s1["mean"]) + f" | {1 / NG:.3f} each |",
          f"| top group fraction (mean) | {s1['maxfrac']:.3f} | ~{1 / NG:.3f} |",
          f"| positions with a group above twice its share (> {200 / NG:.0f} %) | "
          f"{100 * s1['dom']:.1f} % | ~0 % |",
          f"| positions whose top group is the most common one | {100 * s1['stable']:.1f} % | ~{100 / NG:.0f} % |",
          f"| energy CV across blocks of a group | {'n/a' if isinstance(cv, list) else f'{cv:.3f}'} | ~0.125 |",
          f"| head rows: top group fraction (mean) | {rowfrac.mean():.3f} | ~{1 / NG:.3f} |",
          "", "## Previews of b blocks (head fraction = b/" + str(nb) + ")", "",
          "| Preview | M | argmax in cands (+ctx) | full top-20 | lost log-prob (nats/pos) |",
          "|---|---|---|---|---|"]
    for k, M, h1c, h20, loss in res:
        L_.append(f"| {k} | {M} | {100 * h1c:.2f} % | {100 * h20:.2f} % | {loss:.4f} |")
    L_ += ["", "## Exact certificate (Cauchy-Schwarz with block norms)", "",
           "| Preview | rows not excluded (mean) | positions certified with <= 1024 cands |",
           "|---|---|---|"]
    for k, (fr, ok) in cert.items():
        L_.append(f"| {k} | {100 * fr:.2f} % | {100 * ok:.1f} % |")
    rep = "\n".join(L_)
    with open(os.path.join(a.out, f"{tag}.struct.md"), "w") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
