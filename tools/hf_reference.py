#!/usr/bin/env python3
"""Greedy reference logits from transformers for a .tbin parity test."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_np import write_ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--out", default="bitnet.ref")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    ids = tok(a.prompt).input_ids
    print("prompt ids:", ",".join(map(str, ids)))

    logits = []
    with torch.no_grad():
        out = model(torch.tensor([ids]), use_cache=True)
        past = out.past_key_values
        logits.extend(out.logits[0].float().numpy())
        nxt = int(np.argmax(logits[-1]))
        gen = []
        for i in range(a.gen):
            gen.append(nxt)
            if i == a.gen - 1:
                break
            out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            lg = out.logits[0, -1].float().numpy()
            logits.append(lg)
            nxt = int(np.argmax(lg))

    vocab = model.config.vocab_size
    write_ref(a.out, vocab, ids, gen, np.stack(logits)[:, :vocab])
    print("ids generados:", ",".join(map(str, gen)))
    print("text:", tok.decode(ids + gen))
    print(f"referencia escrita en {a.out}")


if __name__ == "__main__":
    main()
