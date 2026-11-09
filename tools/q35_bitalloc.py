#!/usr/bin/env python3
"""Per-head state bit allocation from the GDN decay (reverse water-filling on the noise gain)."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_gguf as G


def head_gains(path):
    r = G.Reader(path)
    out = []
    l = 0
    while f"blk.{l}.attn_norm.weight" in r.tensors:
        n = f"blk.{l}.ssm_a"
        if n in r.tensors:
            a = r.floats(n).astype(np.float64)
            dt = r.floats(f"blk.{l}.ssm_dt.bias").astype(np.float64)
            sp = np.where(dt > 20, dt, np.log1p(np.exp(np.minimum(dt, 20))))
            s = np.maximum(-a * sp, 1e-6)
            for h, x in enumerate(s):
                out.append((l, h, float(x), float(1.0 / np.expm1(2 * x))))
        l += 1
    return out


def allocate(gains, bbar):
    g = np.array([x[3] for x in gains])
    gm = np.exp(np.mean(np.log(g)))
    return bbar + 0.5 * np.log2(g / gm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--bits", type=float, default=8.0, help="mean bits per state value")
    ap.add_argument("--hi", type=float, default=12.0, help="threshold for 16-bit heads")
    a = ap.parse_args()
    gains = head_gains(a.model)
    if not gains:
        sys.exit("no GDN layers")
    b = allocate(gains, a.bits)
    frac = float(np.mean(b >= a.hi))
    s = np.array([x[2] for x in gains])
    print(f"{len(gains)} v-heads; decay s = -a*softplus(dt): median {np.median(s):.3g}, "
          f"min {s.min():.3g}, max {s.max():.3g}")
    print(f"b_i = {a.bits:g} + 0.5 log2(G_i / GM(G)), G_i = 1/expm1(2 s_i)")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"  p{q:<3d} b = {np.percentile(b, q):6.2f}")
    print(f"heads with b >= {a.hi:g}: {100 * frac:.1f} %  ->  q35 --state int8|ro8 --hi {frac:.3f}")
    order = np.argsort(-b)[:8]
    print("top heads (layer, head, s, b): " + ", ".join(
        f"({gains[i][0]}, {gains[i][1]}, {gains[i][2]:.3g}, {b[i]:.1f})" for i in order))


if __name__ == "__main__":
    main()
