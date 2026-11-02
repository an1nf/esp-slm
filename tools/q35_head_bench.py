"""Truncated output head: probability mass lost by vocab order."""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_cmp as C
import q35_gguf as G
from q35_spec_bench import PROMPTS, CHAT

EXTRA = {
    "story": CHAT.format("Write a short story (about 150 words) about a robot that learns to paint."),
    "recipe": CHAT.format("Give me a simple pancake recipe with ingredients and numbered steps."),
}
KS = tuple(int(k) for k in os.environ.get("Q35_HEAD_KS", "8192,16384,32768,65536,131072").split(","))


def vocab_info(path):
    r = G.Reader(path)
    if "tokenizer.ggml.tokens" not in r.kv:
        V = int(r.tensors["token_embd.weight"][1][1])
        return V, np.zeros(0, int), np.zeros(0, int), np.arange(V), 0.0
    toks = r.kv["tokenizer.ggml.tokens"]
    V = len(toks)
    types = np.asarray(r.kv.get("tokenizer.ggml.token_type", [1] * V))
    merges = r.kv.get("tokenizer.ggml.merges", [])
    special = np.where((types == 3) | (types == 4))[0]
    tok2id = {t: i for i, t in enumerate(toks)}
    BIG = 1 << 40
    rank = np.full(V, BIG, np.int64)
    for k, m in enumerate(merges):
        a, _, b = m.partition(" ")
        i = tok2id.get(a + b)
        if i is not None and rank[i] == BIG:
            rank[i] = k
    rank[special] = -2
    base = np.where(rank == BIG)[0]
    rank[base] = -1
    order = np.argsort(rank, kind="stable")
    merged = rank >= 0
    ids = np.arange(V)[merged]
    rho = float(np.corrcoef(np.argsort(np.argsort(ids)), np.argsort(np.argsort(rank[merged])))[0, 1])
    return V, special, base, order, rho


def sets(V, special, order_ids, K, ctx=None):
    m = np.zeros(V, bool)
    m[order_ids[:K]] = True
    m[special] = True
    if ctx is not None:
        m[ctx] = True
    return m


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
    V, special, base, order, rho = vocab_info(a.model)
    print(f"vocab {V}: {len(special)} special, {len(base)} base; id/merge rank corr {rho:.3f}",
          flush=True)
    orders = {"id": np.arange(V), "merge": order}
    texts = dict(PROMPTS)
    texts.update(EXTRA)
    keys = [(o, k, c) for o in orders for k in KS for c in (False, True)]
    res = {x: [] for x in keys}
    lost = {x: [] for x in keys}
    tops_i, tops_p, amax, pos_ctx = [], [], [], []
    npos = 0
    gens = {}
    for name, text in texts.items():
        tok = os.path.join(a.out, f"{tag}.{name}.ids.bin")
        r = subprocess.run([a.llama, a.model, tok, "--text", text, "--tokens-only", "1"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        ids, _ = C.load(tok)
        ids = [int(t) for t in ids]
        print(f"{name}: generating", flush=True)
        r = subprocess.run([a.q35, "spec", a.model, "--ids", ",".join(map(str, ids)),
                            "--gen", str(a.gen), "--ctx", str(len(ids) + a.gen + 40),
                            "--drafter", "mix", "--gamma", "8", "--stop", "--print"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout + r.stderr[-2000:])
        gen = [int(t) for t in re.search(r"^ids: (.*)$", r.stdout, re.M).group(1).split(",")]
        gens[name] = r.stdout[r.stdout.find("\n", r.stdout.find("ids: ")) + 1:].strip()
        seq = ids + gen
        dump = os.path.join(a.out, f"{tag}.{name}.logits.bin")
        print(f"{name}: dumping {len(seq)} positions", flush=True)
        r = subprocess.run([a.q35, "dump", a.model, dump, "--ids", ",".join(map(str, seq)),
                            "--ctx", str(len(seq) + 8), "--chunk", "16"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        _, lg = C.load(dump)
        n0 = len(ids)
        masks = {(o, k): sets(V, special, orders[o], k) for o in orders for k in KS}
        for p in range(n0 - 1, len(seq) - 1):
            x = lg[p].astype(np.float64)
            pr = np.exp(x - x.max())
            pr /= pr.sum()
            full = int(x.argmax())
            ctx = np.unique(np.asarray(seq[:p + 1]))
            top = np.argpartition(-pr, 256)[:256]
            top = top[np.argsort(-pr[top])]
            tops_i.append(top.astype(np.int32))
            tops_p.append(pr[top].astype(np.float32))
            amax.append(full)
            pos_ctx.append(ctx.astype(np.int32))
            for (o, k, c) in keys:
                m = masks[(o, k)]
                if c:
                    m = m.copy()
                    m[ctx] = True
                res[(o, k, c)].append(bool(m[full]))
                lost[(o, k, c)].append(-np.log(max(pr[m].sum(), 1e-300)))
            npos += 1
        os.remove(dump)
    np.savez_compressed(os.path.join(a.out, f"{tag}.head.npz"), top_ids=np.stack(tops_i),
                        top_p=np.stack(tops_p), argmax=np.asarray(amax, np.int32),
                        ctx=np.asarray(pos_ctx, dtype=object), order=order.astype(np.int32),
                        special=special.astype(np.int32), base=base.astype(np.int32))
    L = [f"# Truncated head, {tag}", "",
         f"{npos} generated positions (greedy, {len(texts)} prompts, up to {a.gen} tokens, stop at "
         f"end of turn). Vocab {V}: {len(special)} special and {len(base)} base tokens always kept. "
         f"Rank correlation between id and BPE merge order: {rho:.3f}.", "",
         "S = first K tokens in the given order (plus specials), optionally plus all context "
         "tokens. -log p(S) = probability mass lost by truncation, in nats.", "",
         "| Order | K | argmax equal | -log p(S) | argmax equal (+ctx) | -log p(S) (+ctx) |",
         "|---|---|---|---|---|---|"]
    for o in orders:
        for k in KS:
            f, c = np.asarray(res[(o, k, False)]), np.asarray(res[(o, k, True)])
            lf, lc = np.asarray(lost[(o, k, False)]), np.asarray(lost[(o, k, True)])
            kk = f"{k // 1024}K" if k >= 1024 else str(k)
            L.append(f"| {o} | {kk} | {100 * f.mean():.1f} % | {lf.mean():.2e} | "
                     f"{100 * c.mean():.1f} % | {lc.mean():.2e} |")
    L += ["", "## Generated text", ""]
    for k, v in gens.items():
        L += [f"### {k}", "", "```", v, "```", ""]
    rep = "\n".join(L)
    with open(os.path.join(a.out, f"{tag}.head.md"), "w") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
