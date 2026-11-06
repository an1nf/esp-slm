#!/usr/bin/env python3
"""Write a full-size synthetic Ternary Bonsai 2 GGUF for timing runs."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_gguf as G

F = np.float32
A = "qwen35"
CFG = {
    "2B": dict(E=2048, Fd=6144, L=24, nh=8, nkv=2, hd=256, Hk=16, Hv=16, d=128, tied=True),
    "4B": dict(E=2560, Fd=9216, L=32, nh=16, nkv=4, hd=256, Hk=16, Hv=32, d=128, tied=True),
    "9B": dict(E=4096, Fd=12288, L=32, nh=16, nkv=4, hd=256, Hk=16, Hv=32, d=128, tied=False),
}
V = 248320


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("size", choices=sorted(CFG))
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=0, help="limit the number of layers")
    a = ap.parse_args()
    c = CFG[a.size]
    rng = np.random.default_rng(0)
    E, Fd, hd = c["E"], c["Fd"], c["hd"]
    L = a.layers or c["L"]
    Hk, Hv, dk = c["Hk"], c["Hv"], c["d"]
    kd, vd = Hk * dk, Hv * dk
    cd = 2 * kd + vd
    w = G.Writer()
    w.add("general.architecture", G.V_STR, A)
    w.add("general.alignment", G.V_U32, 32)
    for k, v in [("embedding_length", E), ("block_count", L), ("feed_forward_length", Fd),
                 ("attention.head_count", c["nh"]), ("attention.head_count_kv", c["nkv"]),
                 ("attention.key_length", hd), ("attention.value_length", hd),
                 ("rope.dimension_count", 64), ("ssm.conv_kernel", 4), ("ssm.state_size", dk),
                 ("ssm.time_step_rank", Hv), ("ssm.group_count", Hk), ("ssm.inner_size", vd),
                 ("full_attention_interval", 4), ("context_length", 262144), ("vocab_size", V)]:
        w.add(f"{A}.{k}", G.V_U32, v)
    w.add(f"{A}.rope.freq_base", G.V_F32, 10000000.0)
    w.add(f"{A}.attention.layer_norm_rms_epsilon", G.V_F32, 1e-6)
    w.add(f"{A}.rope.dimension_sections", G.V_ARR, (G.V_I32, [11, 11, 10, 0]))
    w.add("tokenizer.ggml.model", G.V_STR, "no_vocab")

    def tern(name, rows, cols):
        nb = rows * (cols // 128)
        blk = rng.integers(0, 256, (nb, 28), dtype=np.uint8)
        d = np.full(nb, 1.0 / np.sqrt(cols), np.float32).astype(np.float16)
        blk[:, 26:28] = d.view(np.uint8).reshape(nb, 2)
        w.tensor(name, G.T_PTQ1_0, [cols, rows], blk.tobytes())

    def f32(name, shape, x):
        w.tensor(name, G.T_F32, list(shape), np.asarray(x, F).tobytes())

    tern("token_embd.weight", V, E)
    f32("output_norm.weight", [E], np.ones(E))
    if not c["tied"]:
        tern("output.weight", V, E)
    for l in range(L):
        p = f"blk.{l}."
        f32(p + "attn_norm.weight", [E], np.ones(E))
        f32(p + "post_attention_norm.weight", [E], np.ones(E))
        if (l + 1) % 4 != 0:
            tern(p + "attn_qkv.weight", cd, E)
            tern(p + "attn_gate.weight", vd, E)
            f32(p + "ssm_beta.weight", [E, Hv], rng.normal(0, 0.02, (Hv, E)))
            f32(p + "ssm_alpha.weight", [E, Hv], rng.normal(0, 0.02, (Hv, E)))
            tern(p + "ssm_out.weight", E, vd)
            f32(p + "ssm_conv1d.weight", [4, cd], rng.normal(0, 0.3, (cd, 4)))
            f32(p + "ssm_a", [Hv], -rng.uniform(0.3, 1.5, Hv))
            f32(p + "ssm_dt.bias", [Hv], rng.normal(0, 0.5, Hv))
            f32(p + "ssm_norm.weight", [dk], np.ones(dk))
        else:
            tern(p + "attn_q.weight", 2 * c["nh"] * hd, E)
            tern(p + "attn_k.weight", c["nkv"] * hd, E)
            tern(p + "attn_v.weight", c["nkv"] * hd, E)
            tern(p + "attn_output.weight", E, c["nh"] * hd)
            f32(p + "attn_q_norm.weight", [hd], np.ones(hd))
            f32(p + "attn_k_norm.weight", [hd], np.ones(hd))
        tern(p + "ffn_gate.weight", Fd, E)
        tern(p + "ffn_up.weight", Fd, E)
        tern(p + "ffn_down.weight", E, Fd)
    w.write(a.out)


if __name__ == "__main__":
    main()
