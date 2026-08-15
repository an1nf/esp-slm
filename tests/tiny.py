#!/usr/bin/env python3
"""Tiny .tbin models and references for tests."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from ref_np import KV_F32, KV_I8, RefModel, greedy_reference, write_ref
from tbin import FMT_T2, FMT_T3, HEAD_F32, HEAD_I8, random_params, write_tbin

TINY = dict(dim=64, n_layers=3, n_heads=4, n_kv=2, head_dim=16, inter=96,
            vocab=300, max_seq=64, rope_theta=10000.0, eps=1e-5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="build")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--gen", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    params = random_params(TINY, args.seed)
    write_tbin(os.path.join(args.out, "tiny_t2.tbin"), TINY, params, FMT_T2)
    write_tbin(os.path.join(args.out, "tiny_t3.tbin"), TINY, params, FMT_T3)
    write_tbin(os.path.join(args.out, "tiny_t3_h8.tbin"), TINY, params, FMT_T3,
               HEAD_I8)

    rng = np.random.default_rng(args.seed + 1)
    prompt = rng.integers(0, TINY["vocab"], size=args.prompt_len).tolist()
    for name, hf, kf in (("tiny.ref", HEAD_F32, KV_F32),
                         ("tiny_kv8.ref", HEAD_F32, KV_I8),
                         ("tiny_h8kv8.ref", HEAD_I8, KV_I8)):
        g, lg = greedy_reference(RefModel(TINY, params, hf, kf), prompt, args.gen)
        write_ref(os.path.join(args.out, name), TINY["vocab"], prompt, g, lg)
        if name == "tiny.ref":
            gen, logits = g, lg

    top2 = np.sort(logits, axis=1)[:, -2:]
    margin = float(np.min(top2[:, 1] - top2[:, 0]))
    print(f"tiny model: {TINY['n_layers']} layers, dim {TINY['dim']}, "
          f"vocab {TINY['vocab']}; prompt {len(prompt)}, generated {len(gen)}; "
          f"min top1-top2 margin {margin:.3g}; max logit {np.abs(logits).max():.3g}")
    np.save(os.path.join(args.out, "tiny_params_seed.npy"), np.array([args.seed]))


if __name__ == "__main__":
    main()
