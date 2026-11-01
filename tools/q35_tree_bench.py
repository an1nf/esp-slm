"""Tree verification and adjacency drafts on real prompts."""
import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_cmp as C
import p4_twin as TW
from q35_spec_bench import CHAT, PROMPTS

WARM = {
    "story": CHAT.format("Write a short story (about 150 words) about a robot that learns to paint."),
    "recipe": CHAT.format("Give me a simple pancake recipe with ingredients and numbered steps."),
    "explain": CHAT.format("Explain what photosynthesis is and why it matters for life on Earth."),
    "advice": CHAT.format("Give me five practical tips to sleep better, with a short explanation for each."),
    "email": CHAT.format("Write a short, polite email asking my professor for a meeting next week."),
    "code2": CHAT.format("Write a Python function that checks whether a string is a palindrome, with comments."),
}

SAMPLING = ["--temp", "0.7", "--top-k", "20", "--top-p", "0.8", "--presence", "1.5", "--seed", "1"]


def parse(out):
    m = re.search(r"spec: (\d+) tokens in (\d+) passes \(([\d.]+) tokens/pass\).*mean T ([\d.]+)", out)
    t = re.search(r"decode ([\d.]+) s", out)
    src = re.search(r"by source:\n?(.*)", out)
    hist = re.search(r"accepted drafts per pass:(.*)", out)
    return dict(tokens=int(m.group(1)), passes=int(m.group(2)), tau=float(m.group(3)),
                T=float(m.group(4)), secs=float(t.group(1)),
                src=(src.group(1).strip() if src else ""), hist=(hist.group(1).strip() if hist else ""),
                check=("CHECK OK" in out) if "CHECK" in out else None)


def p4_time(size, lanes, sd, core):
    ro = core == "ro8"
    v = TW.Variant(size, max(1, round(lanes)), 1024 if size == "2B" else 512,
                   1 if ro or size == "4B" else 2, 2 if size == "2B" else 1, core=core,
                   wy_rank=(32 if size == "2B" else 4) if ro else 8,
                   wy_flush=(16 if size == "2B" else 2) if ro else 4)
    return TW.run_pass(v, TW.replace(TW.P4(), sd_bw=sd))["time"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--q35", default="build/q35")
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--trees", default="4,8,16,24,32")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--sampling", action="store_true")
    ap.add_argument("--adj", action="store_true",
                    help="add adjacency drafts, warmed on separate prompts")
    ap.add_argument("--only-adj", action="store_true",
                    help="only none, chain and adjacency configs")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--suffix", default="", help="report file suffix, e.g. .adj")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0]
    log = open(os.path.join(a.out, f"{tag}.tree{a.suffix}.log"), "w")
    info = subprocess.run([a.q35, "info", a.model], capture_output=True, text=True).stdout
    E = int(re.search(r"embd (\d+)", info).group(1))
    size = {2048: "2B", 2560: "4B", 4096: "9B"}.get(E, "2B")
    trees = [int(x) for x in a.trees.split(",") if x]
    runs = [("none", [], "greedy"), ("chain mix g=8", ["--drafter", "mix", "--gamma", "8"], "greedy")]
    for N in ([] if a.only_adj else trees):
        extra = ["--tree", str(N), "--gamma", str(a.depth)]
        if N == 16:
            extra += ["--check"]
        runs.append((f"tree {N}", extra, "greedy"))
    if a.sampling and not a.only_adj:
        runs += [("chain mix g=8", ["--drafter", "mix", "--gamma", "8"] + SAMPLING, "sampled"),
                 ("tree 16", ["--tree", "16", "--gamma", str(a.depth), "--check"] + SAMPLING, "sampled"),
                 ("tree 32", ["--tree", "32", "--gamma", str(a.depth)] + SAMPLING, "sampled"),
                 ("tree 16", ["--tree", "16", "--gamma", str(a.depth), "--keyed", "0"] + SAMPLING,
                  "sampled, unkeyed")]
    if a.adj or a.only_adj:
        adjf = os.path.join(a.out, f"{tag}.adj")
        if os.path.exists(adjf):
            os.remove(adjf)
        for name, text in WARM.items():
            tok = os.path.join(a.out, f"{tag}.warm.{name}.ids.bin")
            r = subprocess.run([a.llama, a.model, tok, "--text", text, "--tokens-only", "1"],
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stderr[-2000:])
            ids, _ = C.load(tok)
            ids_s = ",".join(str(int(t)) for t in ids)
            print(f"warming adjacency: {name}", flush=True)
            r = subprocess.run([a.q35, "spec", a.model, "--ids", ids_s, "--gen", "128",
                                "--ctx", str(len(ids) + 128 + 80), "--tree", "16", "--gamma", str(a.depth),
                                "--adj", "1", "--adj-file", adjf, "--stop"], capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stdout + r.stderr[-1500:])
        warm = ["--adj", "1", "--adj-file", adjf, "--adj-ro", "1"]
        D = ["--gamma", str(a.depth)]
        if a.only_adj:
            runs.append(("tree 16", ["--tree", "16"] + D, "greedy"))
        runs += [("tree 16 + adj", ["--tree", "16"] + D + warm + ["--check"], "greedy"),
                 ("tree 32 + adj", ["--tree", "32"] + D + warm, "greedy"),
                 ("tree 16 + cold adj", ["--tree", "16"] + D + ["--adj", "1"], "greedy")]
        if a.sampling:
            if a.only_adj:
                runs += [("chain mix g=8", ["--drafter", "mix", "--gamma", "8"] + SAMPLING, "sampled"),
                         ("tree 16", ["--tree", "16"] + D + ["--check"] + SAMPLING, "sampled"),
                         ("tree 16", ["--tree", "16"] + D + ["--keyed", "0"] + SAMPLING,
                          "sampled, unkeyed")]
            runs += [("tree 16 + adj", ["--tree", "16"] + D + warm + ["--check"] + SAMPLING,
                      "sampled")]
    rows = []
    for name in a.prompts.split(","):
        text = PROMPTS[name]
        tok = os.path.join(a.out, f"{tag}.{name}.ids.bin")
        r = subprocess.run([a.llama, a.model, tok, "--text", text, "--tokens-only", "1"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        ids, _ = C.load(tok)
        ids_s = ",".join(str(int(t)) for t in ids)
        for label, extra, mode in runs:
            lanes = 1
            if "--tree" in extra:
                lanes = int(extra[extra.index("--tree") + 1])
            if "--gamma" in extra:
                lanes = max(lanes, int(extra[extra.index("--gamma") + 1]) + 1)
            cmd = [a.q35, "spec", a.model, "--ids", ids_s, "--gen", str(a.gen),
                   "--ctx", str(len(ids) + a.gen + lanes + 40)]
            cmd += extra if extra else ["--drafter", "none", "--gamma", "1"]
            print(f"{name}: {label} ({mode})", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            log.write(f"$ q35 spec {name} {label} {mode}: {' '.join(extra)}\n{r.stdout}\n{r.stderr[-1500:]}\n")
            log.flush()
            if r.returncode:
                sys.exit(f"q35 spec failed ({name}, {label}):\n{r.stdout}\n{r.stderr[-1500:]}")
            p = parse(r.stdout)
            p.update(name=name, label=label, mode=mode, wall=time.time() - t0)
            for core in ("seq", "ro8"):
                for sd in (40, 80):
                    p[f"{core}_{sd}"] = p4_time(size, p["T"], sd, core) / p["tau"]
            print(f"   tau {p['tau']:.2f}, nodes {p['T']:.1f}, PC {p['secs'] / max(1, p['passes']):.2f} s/pass, "
                  f"P4 ro8 SD40 {p['ro8_40']:.1f} s/token", flush=True)
            rows.append(p)
    L = [f"# Tree verification, {tag} ({size})", "",
         f"{a.gen} tokens per prompt, tree depth <= {a.depth}. tau = tokens per pass (incl. bonus "
         "token); nodes = tokens evaluated per pass. P4 s/token from p4_twin (state and KV "
         + ("f16, ctx 1024" if size == "2B" else "int8, ctx 512") + "), seq and ro8 cores.", "",
         "## Summary", "",
         "| Config | mode | tau | nodes | P4 seq SD 40 | P4 ro8 SD 40 | P4 ro8 SD 80 |",
         "|---|---|---|---|---|---|---|"]
    seen = []
    for p in rows:
        k = (p["label"], p["mode"])
        if k not in seen:
            seen.append(k)
    for k in seen:
        g = [p for p in rows if (p["label"], p["mode"]) == k]
        tok = sum(p["tokens"] for p in g)
        pas = sum(p["passes"] for p in g)
        tau = tok / pas
        lanes = sum(p["T"] * p["passes"] for p in g) / pas
        L.append(f"| {k[0]} | {k[1]} | {tau:.2f} | {lanes:.1f} | "
                 f"{p4_time(size, lanes, 40, 'seq') / tau:.1f} | {p4_time(size, lanes, 40, 'ro8') / tau:.1f} | "
                 f"{p4_time(size, lanes, 80, 'ro8') / tau:.1f} |")
    L += ["", "## Per prompt", "",
          "| Prompt | Config | mode | tau | nodes | accepted per pass (k:count) | PC s/pass | check |",
          "|---|---|---|---|---|---|---|---|"]
    for p in rows:
        chk = "" if p["check"] is None else ("OK" if p["check"] else "FAIL")
        L.append(f"| {p['name']} | {p['label']} | {p['mode']} | {p['tau']:.2f} | {p['T']:.1f} | "
                 f"{p['hist']} | {p['secs'] / max(1, p['passes']):.2f} | {chk} |")
    L += ["", "## Tree sources (accepted/proposed, parent accepted)", ""]
    for p in rows:
        if p["label"].startswith("tree"):
            L.append(f"- {p['name']}, {p['label']} ({p['mode']}): {p['src']}")
    rep = "\n".join(L)
    with open(os.path.join(a.out, f"{tag}.tree{a.suffix}.md"), "w") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
