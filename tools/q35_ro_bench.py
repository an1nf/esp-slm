"""KL of GDN state and KV formats on real prompts."""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_cmp as C
from q35_head_bench import EXTRA
from q35_spec_bench import PROMPTS

CANDS = "f16,bf16,int8,ro32:16:2,ro16:16:2,ro8:8:2,ro8:16:2,ro8:32:2,ro8:16:16"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--q35", default="build/q35")
    ap.add_argument("--gen", type=int, default=160)
    ap.add_argument("--cands", default=CANDS)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0]
    log = open(os.path.join(a.out, f"{tag}.ro.log"), "w")
    texts = dict(PROMPTS)
    texts.update(EXTRA)
    agg = {}
    order = []
    for name, text in texts.items():
        if a.llama == "none":
            ids = [(7 * k + len(name)) % 97 + 1 for k in range(12)]
        else:
            tok = os.path.join(a.out, f"{tag}.{name}.ids.bin")
            r = subprocess.run([a.llama, a.model, tok, "--text", text, "--tokens-only", "1"],
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stderr[-2000:])
            ids = [int(t) for t in C.load(tok)[0]]
        print(f"{name}: generating", flush=True)
        r = subprocess.run([a.q35, "spec", a.model, "--ids", ",".join(map(str, ids)), "--gen", str(a.gen),
                            "--ctx", str(len(ids) + a.gen + 40), "--drafter", "mix", "--gamma", "8", "--stop"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout + r.stderr[-2000:])
        gen = [int(t) for t in re.search(r"^ids: (.*)$", r.stdout, re.M).group(1).split(",")]
        seq = ids + gen
        print(f"{name}: {len(seq)} positions, comparing state formats", flush=True)
        r = subprocess.run([a.q35, "cmpstate", a.model, "--cands", a.cands, "--ids", ",".join(map(str, seq))],
                           capture_output=True, text=True)
        log.write(f"## {name} ({len(seq)} positions)\n{r.stdout}\n{r.stderr[-1500:]}\n")
        log.flush()
        if r.returncode:
            sys.exit(r.stdout + r.stderr[-2000:])
        n = len(seq)
        n2 = n - n // 2
        for line in r.stdout.splitlines():
            if not line.startswith("| ") or line.startswith("| candidate"):
                continue
            c = [x.strip() for x in line.strip("|").split("|")]
            key = c[0]
            if key not in agg:
                agg[key] = [0, 0, 0.0, 0.0, 0, 0]
                order.append(key)
            same = int(c[1].split("/")[0])
            x = agg[key]
            x[0] += n
            x[1] += same
            x[2] += float(c[2]) * n
            x[3] += float(c[5]) * n2
            x[4] += n2
            x[5] += int(c[6])
            print(f"   {line}", flush=True)
    L = [f"# GDN state formats, {tag}", "",
         f"{len(texts)} prompts, greedy, up to {a.gen} generated tokens, teacher-forced. Reference: "
         "exact f32 state. KL(exact || candidate) per position, pooled over all positions.", "",
         "| Candidate | argmax equal | KL mean | KL mean 2nd half | materializations |",
         "|---|---|---|---|---|"]
    for k in order:
        n, same, s, s2, n2, mat = agg[k]
        L.append(f"| {k} | {same}/{n} ({100 * same / n:.2f} %) | {s / n:.2e} | {s2 / n2:.2e} | {mat} |")
    rep = "\n".join(L)
    with open(os.path.join(a.out, f"{tag}.ro.md"), "w") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
