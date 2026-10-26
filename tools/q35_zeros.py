"""Zero density of PTQ1_0 weights and entropy coding bounds."""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_gguf as G

ROWS = 4096


def hbin(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def kind(name):
    for k in ("token_embd", "output", "attn_qkv", "attn_gate", "ssm_out", "ssm_alpha", "ssm_beta",
              "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        if k in name:
            return k
    return "otros"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = G.Reader(a.model)
    agg = defaultdict(lambda: [0, 0, 0.0, 0])
    for name, info in r.tensors.items():
        if int(info[0]) != 143:
            continue
        cols, rows = int(info[1][0]), int(info[1][1])
        raw = r.raw(name)[2]
        rb = cols // 128 * 28
        k = kind(name)
        agg[k][3] += 1
        for r0 in range(0, rows, ROWS):
            r1 = min(rows, r0 + ROWS)
            t, _ = G.ptq1_decode(raw[r0 * rb:r1 * rb], r1 - r0, cols)
            zb = (t.reshape(-1, 128) == 0).mean(1)
            agg[k][0] += t.size
            agg[k][1] += int((t == 0).sum())
            agg[k][2] += float(((hbin(zb) + (1 - zb)) * 128 + 7).sum())
        print(f"  {name}", flush=True)
    L = [f"# Zero density, {os.path.basename(a.model)}", "",
         "Rates in bits/weight without the scale (PTQ1_0: 1.625). Savings vs PTQ1_0 include the "
         "fp16 block scale (0.125 bits/weight) in every format.", "",
         "| Tensors | n | weights (M) | z | BITCOS 2-z | entropy (global z) | entropy (block z) | "
         "BITCOS saving | block-entropy saving |",
         "|---|---|---|---|---|---|---|---|---|"]
    tot = [0, 0, 0.0]
    for k, (n, nz, hb, nt) in sorted(agg.items(), key=lambda x: -x[1][0]):
        z = nz / n
        bit = 2 - z
        hg = float(hbin(z) + (1 - z))
        hpb = hb / n
        base = 1.625 + 0.125
        L.append(f"| {k} | {nt} | {n / 1e6:.1f} | {z:.4f} | {bit:.3f} | {hg:.3f} | {hpb:.3f} | "
                 f"{100 * (1 - (min(bit, 1.625) + 0.125) / base):.1f} % | {100 * (1 - (hpb + 0.125) / base):.1f} % |")
        tot[0] += n
        tot[1] += nz
        tot[2] += hb
    z = tot[1] / tot[0]
    hpb = tot[2] / tot[0]
    L.append(f"| **all** | | {tot[0] / 1e6:.1f} | {z:.4f} | {2 - z:.3f} | {float(hbin(z) + 1 - z):.3f} | "
             f"{hpb:.3f} | {100 * (1 - (min(2 - z, 1.625) + 0.125) / 1.75):.1f} % | "
             f"{100 * (1 - (hpb + 0.125) / 1.75):.1f} % |")
    rep = "\n".join(L)
    print(rep)
    if a.out:
        with open(a.out, "w") as f:
            f.write(rep + "\n")


if __name__ == "__main__":
    main()
