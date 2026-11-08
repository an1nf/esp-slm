#!/usr/bin/env python3
"""Build an HDS1 suffix-array datastore for `q35 spec --tree N --ds FILE`."""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_cmp as C

SEP = -1


def suffix_array(t: np.ndarray) -> np.ndarray:
    n = len(t)
    if n == 0:
        return np.zeros(0, np.int32)
    rank = t.astype(np.int64) + 1
    k = 1
    while True:
        r2 = np.full(n, -1, np.int64)
        r2[:n - k] = rank[k:]
        sa = np.lexsort((r2, rank))
        a, b = rank[sa], r2[sa]
        diff = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
        new = np.empty(n, np.int64)
        new[sa] = np.concatenate([[0], np.cumsum(diff)])
        rank = new
        if rank.max() == n - 1 or k >= n:
            return sa.astype(np.int32)
        k *= 2


def write_hds(path: str, docs: list, vocab: int = 0) -> int:
    parts = []
    for d in docs:
        parts.append(np.asarray(d, np.int64))
        parts.append(np.array([SEP], np.int64))
    t = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    sa = suffix_array(t)
    with open(path, "wb") as f:
        f.write(b"HDS1")
        f.write(struct.pack("<II", len(t), vocab))
        f.write(t.astype("<i4").tobytes())
        f.write(sa.astype("<i4").tobytes())
    return len(t)


def tokenize(llama: str, model: str, text: str) -> list:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ids.bin")
        r = subprocess.run([llama, model, p, "--text", text, "--tokens-only", "1"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        return [int(x) for x in C.load(p)[0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--ids", action="append", default=[], help="comma-separated token ids")
    ap.add_argument("--ids-file", action="append", default=[], help="Q35R file from llama_logits")
    ap.add_argument("--text", action="append", default=[], help="text file, one document per blank-line block")
    ap.add_argument("--model")
    ap.add_argument("--llama")
    ap.add_argument("--vocab", type=int, default=0)
    a = ap.parse_args()
    docs = [[int(x) for x in s.split(",") if x.strip()] for s in a.ids]
    docs += [[int(x) for x in C.load(p)[0]] for p in a.ids_file]
    for p in a.text:
        if not (a.model and a.llama):
            sys.exit("--text needs --model and --llama")
        for block in open(p, encoding="utf-8").read().split("\n\n"):
            if block.strip():
                docs.append(tokenize(a.llama, a.model, block.strip()))
    n = write_hds(a.out, docs, a.vocab)
    print(f"wrote {a.out}: {len(docs)} docs, {n} tokens")


if __name__ == "__main__":
    main()
