"""Tiny q35 GGUF models and f64 references for tests."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import q35_gguf as G
from q35_ref import Model

F = np.float32
A = "qwen35"


FOLDABLE = ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
            "ffn_gate", "ffn_up", "ffn_down", "ssm_out")


def build(path, seed, n_layer, fused, untied_q8, f16_gate, sect, scale_up, rope_base=10000000.0,
          hadamard=None, eos=None):
    rng = np.random.default_rng(seed)
    E, Fd, V = 256, 384, 512
    nh, nkv, hd, n_rot = 4, 2, 64, 16
    Hk, Hv, dk, K = 2, 4, 32, 4
    dv = dk
    kd, vd = Hk * dk, Hv * dv
    cd = 2 * kd + vd
    w = G.Writer()
    w.add("general.architecture", G.V_STR, A)
    w.add("general.alignment", G.V_U32, 32)
    for k, v in [("embedding_length", E), ("block_count", n_layer), ("feed_forward_length", Fd),
                 ("attention.head_count", nh), ("attention.head_count_kv", nkv),
                 ("attention.key_length", hd), ("attention.value_length", hd),
                 ("rope.dimension_count", n_rot), ("ssm.conv_kernel", K), ("ssm.state_size", dk),
                 ("ssm.time_step_rank", Hv), ("ssm.group_count", Hk), ("ssm.inner_size", vd),
                 ("full_attention_interval", 4)]:
        w.add(f"{A}.{k}", G.V_U32, v)
    w.add(f"{A}.rope.freq_base", G.V_F32, rope_base)
    w.add(f"{A}.attention.layer_norm_rms_epsilon", G.V_F32, 1e-6)
    w.add(f"{A}.rope.dimension_sections", G.V_ARR, (G.V_I32, sect))
    w.add(f"{A}.context_length", G.V_U32, 4096)
    w.add(f"{A}.vocab_size", G.V_U32, V)
    w.add("tokenizer.ggml.model", G.V_STR, "no_vocab")
    if eos is not None:
        w.add("tokenizer.ggml.eos_token_id", G.V_U32, int(eos))

    folded, widths = [], set()

    def fold(name, cols):
        kind = name.split(".")[-2] if name.startswith("blk.") else name.split(".")[0]
        if hadamard and (kind in FOLDABLE or name == "output.weight"):
            folded.append(name)
            widths.add(cols)

    def tern(name, rows, cols, sd=1.0):
        x = rng.normal(0, sd / np.sqrt(cols), (rows, cols)).astype(F)
        w.tensor(name, G.T_PTQ1_0, [cols, rows], G.ptq1_quantize(x))
        fold(name, cols)

    def f32(name, shape, x):
        w.tensor(name, G.T_F32, list(shape), np.asarray(x, F).tobytes())

    def norm(name, n):
        f32(name, [n], 1.0 + 0.1 * rng.normal(size=n))

    tern("token_embd.weight", V, E)
    norm("output_norm.weight", E)
    if untied_q8:
        x = rng.normal(0, 1 / np.sqrt(E), (V, E)).astype(F)
        w.tensor("output.weight", G.T_Q8_0, [E, V], G.q8_quantize_w(x))
        fold("output.weight", E)
    elif hadamard and hadamard["version"] == 1:
        tern("output.weight", V, E)
    for l in range(n_layer):
        p = f"blk.{l}."
        norm(p + "attn_norm.weight", E)
        norm(p + "post_attention_norm.weight", E)
        if (l + 1) % 4 != 0:
            tern(p + "attn_qkv.weight", cd, E)
            if f16_gate and l == 1:
                x = rng.normal(0, 0.05, (vd, E)).astype(np.float16)
                w.tensor(p + "attn_gate.weight", G.T_F16, [E, vd], x.tobytes())
                fold(p + "attn_gate.weight", E)
            else:
                tern(p + "attn_gate.weight", vd, E)
            f32(p + "ssm_beta.weight", [E, Hv], rng.normal(0, 0.05, (Hv, E)))
            f32(p + "ssm_alpha.weight", [E, Hv], rng.normal(0, 0.05, (Hv, E)))
            tern(p + "ssm_out.weight", E, vd)
            f32(p + "ssm_conv1d.weight", [K, cd], rng.normal(0, 0.5, (cd, K)))
            f32(p + "ssm_a", [Hv], -rng.uniform(0.3, 1.5, Hv))
            f32(p + "ssm_dt.bias", [Hv], rng.normal(0, 0.5, Hv))
            norm(p + "ssm_norm.weight", dv)
        else:
            qd, kvd = 2 * nh * hd, nkv * hd
            if fused:
                tern(p + "attn_qkv.weight", qd + 2 * kvd, E)
            else:
                tern(p + "attn_q.weight", qd, E)
                tern(p + "attn_k.weight", kvd, E)
                tern(p + "attn_v.weight", kvd, E)
            tern(p + "attn_output.weight", E, nh * hd)
            norm(p + "attn_q_norm.weight", hd)
            norm(p + "attn_k_norm.weight", hd)
        tern(p + "ffn_gate.weight", Fd, E)
        tern(p + "ffn_up.weight", Fd, E)
        if scale_up and l == 1:
            f32(p + "ffn_up.scale", [1], [0.75])
        tern(p + "ffn_down.weight", E, Fd)
    if hadamard:
        H = "prism.hadamard."
        w.add(H + "version", G.V_U32, hadamard["version"])
        w.add(H + "tied_output", G.V_BOOL, hadamard["version"] == 2)
        w.add(H + "block_size", G.V_U32, hadamard["block"])
        w.add(H + "transform", G.V_STR, "normalized-sylvester-walsh-hadamard")
        w.add(H + "axis", G.V_STR, "input-last-dimension")
        w.add(H + "weight_names", G.V_ARR, (G.V_STR, folded))
        w.add(H + "inverse_weight_names", G.V_ARR, (G.V_STR, ["token_embd.weight"]))
        w.add(H + "gdn_v_grouped", G.V_BOOL, bool(hadamard["grouped"]))
        if hadamard["explicit"]:
            ws = sorted(widths | {E})
            vals = []
            for wd in ws:
                vals += [int(v) for v in rng.choice([-1, 1], wd)]
            w.add(H + "sign_mode", G.V_STR, "explicit")
            w.add(H + "sign_widths", G.V_ARR, (G.V_I32, ws))
            w.add(H + "sign_values", G.V_ARR, (G.V_I32, vals))
        else:
            w.add(H + "sign_mode", G.V_STR, "identity")
    w.write(path)
    return V


def write_ref(model_path, ref_path, ids, kv16):
    m = Model(model_path)
    lg = m.run(ids, kv16)
    with open(ref_path, "wb") as f:
        f.write(b"Q35R" + struct.pack("<III", len(ids), m.V, int(kv16)))
        f.write(np.asarray(ids, np.int32).tobytes())
        f.write(lg.astype(F).tobytes())
    return lg


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    ids = [int(t) for t in np.random.default_rng(7).integers(0, 512, 12)]
    pa = os.path.join(out, "q35_a.gguf")
    pb = os.path.join(out, "q35_b.gguf")
    build(pa, 1, 4, fused=False, untied_q8=False, f16_gate=False, sect=[3, 3, 2, 0], scale_up=False)
    build(pb, 2, 8, fused=True, untied_q8=True, f16_gate=True, sect=[2, 2, 2, 2], scale_up=True,
          rope_base=100.0)
    pd = os.path.join(out, "q35_d.gguf")
    pe = os.path.join(out, "q35_e.gguf")
    build(pd, 3, 4, fused=False, untied_q8=False, f16_gate=False, sect=[3, 3, 2, 0], scale_up=False,
          hadamard=dict(version=2, block=64, explicit=True, grouped=True))
    build(pe, 4, 8, fused=True, untied_q8=False, f16_gate=True, sect=[3, 3, 2, 0], scale_up=False,
          hadamard=dict(version=1, block=128, explicit=False, grouped=False))
    for name, path, kv16 in [("a", pa, True), ("b", pb, True), ("c", pa, False),
                             ("d", pd, True), ("e", pe, True)]:
        lg = write_ref(path, os.path.join(out, f"q35_{name}.ref"), ids, kv16)
        print(f"q35_{name}: logits rms {float(np.sqrt((lg ** 2).mean())):.3f}, "
              f"argmax {lg.argmax(1).tolist()}")


if __name__ == "__main__":
    main()
