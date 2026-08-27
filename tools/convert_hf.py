#!/usr/bin/env python3
"""Convert a Hugging Face BitNet b1.58 checkpoint to .tbin."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_np import hf_unpack, read_safetensors
from tbin import FMT_T2, FMT_T3, HEAD_F32, HEAD_I8, write_tbin

HF_LINEAR = {"q": "self_attn.q_proj", "k": "self_attn.k_proj",
             "v": "self_attn.v_proj", "o": "self_attn.o_proj",
             "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def load_config(d: str) -> dict:
    with open(os.path.join(d, "config.json")) as f:
        hc = json.load(f)
    hc = hc.get("text_config", hc)
    if hc.get("hidden_act", "relu2") != "relu2":
        raise SystemExit(f"activation {hc.get('hidden_act')} not supported (expected relu2)")
    theta = hc.get("rope_theta")
    if theta is None:
        theta = (hc.get("rope_parameters") or {}).get("rope_theta", 10000.0)
    heads = hc["num_attention_heads"]
    return dict(dim=hc["hidden_size"], n_layers=hc["num_hidden_layers"],
                n_heads=heads, n_kv=hc.get("num_key_value_heads", heads),
                head_dim=hc.get("head_dim") or hc["hidden_size"] // heads,
                inter=hc["intermediate_size"], vocab=hc["vocab_size"],
                max_seq=hc.get("max_position_embeddings", 4096),
                rope_theta=float(theta), eps=float(hc.get("rms_norm_eps", 1e-5)),
                tied=bool(hc.get("tie_word_embeddings", True)))


def ternary_linear(t: dict, prefix: str, rows: int, cols: int):
    w, dt = t[prefix + ".weight"]
    sc = t.get(prefix + ".weight_scale")
    if dt == "U8":
        if w.shape[1] != cols or w.shape[0] * 4 < rows:
            raise SystemExit(f"{prefix}: forma empaquetada {w.shape} incompatible")
        wt = hf_unpack(w, rows)
        scale = float(np.asarray(sc[0]).reshape(-1)[0]) if sc else 1.0
        if not sc:
            print(f"warning: {prefix} has no weight_scale, using 1.0")
        return wt, scale
    w = np.asarray(w, np.float32)
    if w.shape != (rows, cols):
        raise SystemExit(f"{prefix}: forma {w.shape}, se esperaba {(rows, cols)}")
    if np.all(np.isin(w, (-1.0, 0.0, 1.0))):
        scale = float(np.asarray(sc[0]).reshape(-1)[0]) if sc else 1.0
        return w.astype(np.int8), scale
    m = max(float(np.mean(np.abs(w), dtype=np.float32)), 1e-5)
    wt = np.clip(np.rint(w / np.float32(m)), -1, 1).astype(np.int8)
    return wt, m


def convert(model_dir: str, out: str, fmt: int, max_seq: int | None,
            head_fmt: int = HEAD_F32) -> dict:
    cfg = load_config(model_dir)
    if max_seq:
        cfg["max_seq"] = min(max_seq, cfg["max_seq"])
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise SystemExit("no hay ficheros .safetensors")
    t = {}
    for f in files:
        t.update(read_safetensors(f))

    def f32(name):
        return np.asarray(t[name][0], np.float32)

    if not cfg["tied"] and "lm_head.weight" in t:
        raise SystemExit("untied output head not supported")
    dim, inter = cfg["dim"], cfg["inter"]
    qdim, kvdim = cfg["n_heads"] * cfg["head_dim"], cfg["n_kv"] * cfg["head_dim"]
    shapes = dict(q=(qdim, dim), k=(kvdim, dim), v=(kvdim, dim), o=(dim, qdim),
                  gate=(inter, dim), up=(inter, dim), down=(dim, inter))
    layers = []
    for i in range(cfg["n_layers"]):
        p = f"model.layers.{i}."
        lay = dict(attn_norm=f32(p + "input_layernorm.weight"),
                   attn_sub_norm=f32(p + "self_attn.attn_sub_norm.weight"),
                   ffn_norm=f32(p + "post_attention_layernorm.weight"),
                   ffn_sub_norm=f32(p + "mlp.ffn_sub_norm.weight"))
        for k, hf in HF_LINEAR.items():
            lay[k] = ternary_linear(t, p + hf, *shapes[k])
        layers.append(lay)
    params = dict(tok_emb=f32("model.embed_tokens.weight"),
                  final_norm=f32("model.norm.weight"), layers=layers)
    write_tbin(out, cfg, params, fmt, head_fmt)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--fmt", choices=("t2", "t3"), default="t2")
    ap.add_argument("--head", choices=("f32", "int8"), default="f32")
    ap.add_argument("--max-seq", type=int, default=2048)
    a = ap.parse_args()
    cfg = convert(a.model_dir, a.out, FMT_T2 if a.fmt == "t2" else FMT_T3, a.max_seq,
                  HEAD_I8 if a.head == "int8" else HEAD_F32)
    print(f"wrote {a.out}: {cfg['n_layers']} layers, dim {cfg['dim']}, "
          f"vocab {cfg['vocab']}, format {a.fmt}, head {a.head}, "
          f"max_seq {cfg['max_seq']}, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
