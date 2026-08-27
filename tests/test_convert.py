#!/usr/bin/env python3
"""HF to .tbin converter round trip."""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from convert_hf import HF_LINEAR, convert
from safetensors_np import bf16_to_f32, f32_to_bf16, hf_pack, hf_unpack, write_safetensors
from tbin import FMT_T2, FMT_T3, HEAD_F32, HEAD_I8, random_params, write_tbin
from tiny import TINY


def bf16_round(x):
    return bf16_to_f32(f32_to_bf16(x)).reshape(np.shape(x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="build")
    a = ap.parse_args()
    d = os.path.join(a.out, "fake_hf")
    os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(7)
    for rows in (1, 3, 4, 6, 13):
        w = rng.integers(-1, 2, size=(rows, 9)).astype(np.int8)
        assert np.array_equal(hf_unpack(hf_pack(w), rows), w), rows

    p = random_params(TINY, seed=3)
    p["tok_emb"] = bf16_round(p["tok_emb"])
    p["final_norm"] = bf16_round(p["final_norm"])
    for lay in p["layers"]:
        for k in ("attn_norm", "attn_sub_norm", "ffn_norm", "ffn_sub_norm"):
            lay[k] = bf16_round(lay[k])
        for k in HF_LINEAR:
            w, s = lay[k]
            lay[k] = (w, float(bf16_round(np.float32(s))))

    cfg = dict(TINY)
    hf_cfg = dict(architectures=["BitNetForCausalLM"], model_type="bitnet",
                  hidden_act="relu2", hidden_size=cfg["dim"],
                  num_hidden_layers=cfg["n_layers"],
                  num_attention_heads=cfg["n_heads"],
                  num_key_value_heads=cfg["n_kv"],
                  intermediate_size=cfg["inter"], vocab_size=cfg["vocab"],
                  max_position_embeddings=cfg["max_seq"],
                  rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["eps"],
                  tie_word_embeddings=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(hf_cfg, f)

    tensors = {"model.embed_tokens.weight": (p["tok_emb"], "BF16"),
               "model.norm.weight": (p["final_norm"], "BF16")}
    for i, lay in enumerate(p["layers"]):
        pre = f"model.layers.{i}."
        tensors[pre + "input_layernorm.weight"] = (lay["attn_norm"], "BF16")
        tensors[pre + "post_attention_layernorm.weight"] = (lay["ffn_norm"], "BF16")
        tensors[pre + "self_attn.attn_sub_norm.weight"] = (lay["attn_sub_norm"], "BF16")
        tensors[pre + "mlp.ffn_sub_norm.weight"] = (lay["ffn_sub_norm"], "BF16")
        for k, hf in HF_LINEAR.items():
            w, s = lay[k]
            tensors[pre + hf + ".weight"] = (hf_pack(w), "U8")
            tensors[pre + hf + ".weight_scale"] = (np.array([s], np.float32), "BF16")
    write_safetensors(os.path.join(d, "model.safetensors"), tensors)

    ok = True
    for fmt, hf, tag in ((FMT_T2, HEAD_F32, "t2"), (FMT_T3, HEAD_F32, "t3"),
                         (FMT_T3, HEAD_I8, "t3, int8 head")):
        expect = os.path.join(a.out, f"fake_expected_{fmt}_{hf}.tbin")
        got = os.path.join(a.out, f"fake_converted_{fmt}_{hf}.tbin")
        write_tbin(expect, cfg, p, fmt, hf)
        convert(d, got, fmt, None, hf)
        same = filecmp.cmp(expect, got, shallow=False)
        print(f"HF -> .tbin converter ({tag}): {'byte-exact' if same else 'DIFFERENT'}")
        ok &= same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
