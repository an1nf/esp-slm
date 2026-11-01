"""Interactive chat on q35 spec with a P4 time estimate per reply."""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_cmp as C
import p4_twin as TW

THINK_OFF = "<think>\n\n</think>\n\n"


def prompt(history, user):
    s = ""
    for u, a in history:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{THINK_OFF}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--q35", default="build/q35")
    ap.add_argument("--gen", type=int, default=192)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--tree", type=int, default=0,
                    help="> 0: N-node tree verification, depth <= --gamma")
    ap.add_argument("--adj-file", default="",
                    help="with --tree: persistent adjacency table file")
    ap.add_argument("--turns", type=int, default=3, help="history turns kept")
    ap.add_argument("--sd", type=float, default=40.0, help="SD MB/s for the P4 estimate")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--presence", type=float, default=1.5)
    a = ap.parse_args()
    info = subprocess.run([a.q35, "info", a.model], capture_output=True, text=True).stdout
    E = int(re.search(r"embd (\d+)", info).group(1))
    size = {2048: "2B", 2560: "4B", 4096: "9B"}.get(E, "2B")
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.model)), ".q35_chat_ids.bin")
    history = []
    print(f"q35: Ternary Bonsai 2 {size}. Type and press Enter ('exit' to quit, '/new' to reset).")
    while True:
        try:
            user = input("\n> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        if user == "/new":
            history = []
            print("(new conversation)")
            continue
        text = prompt(history[-a.turns:], user)
        r = subprocess.run([a.llama, a.model, tmp, "--text", text, "--tokens-only", "1"],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-1000:])
            continue
        ids, _ = C.load(tmp)
        print(f"\n(prefill {len(ids)} tokens)\n", flush=True)
        r = subprocess.run([a.q35, "spec", a.model, "--ids", ",".join(str(int(t)) for t in ids),
                            "--gen", str(a.gen),
                            "--ctx", str(len(ids) + a.gen + max(a.gamma, a.tree) + 8),
                            *(["--tree", str(a.tree)] if a.tree > 0 else ["--drafter", "mix"]),
                            *(["--adj", "1", "--adj-file", a.adj_file] if a.tree > 0 and a.adj_file else []),
                            "--gamma", str(a.gamma), "--stop", "--print",
                            "--stream", "--temp", str(a.temp), "--top-k", str(a.top_k),
                            "--top-p", str(a.top_p), "--presence", str(a.presence),
                            "--seed", str(1 + len(history))], stdout=subprocess.PIPE, text=True)
        if r.returncode:
            print(r.stdout)
            continue
        out = r.stdout
        reply = out[out.find("\n", out.find("ids: ")) + 1:]
        reply = reply.replace("<|im_end|>", "").strip()
        m = re.search(r"spec: (\d+) tokens in (\d+) passes \(([\d.]+) tokens/pass\).*mean T ([\d.]+)", out)
        t = re.search(r"\(([\d.]+) s/token\)", out)
        if m and t:
            tau, T = float(m.group(3)), float(m.group(4))
            sb = 2 if size == "2B" else 1
            ctx = min(1024 if size == "2B" else 512, max(64, len(ids) + int(m.group(1))))
            p4 = TW.run_pass(TW.Variant(size, max(1, round(T)), ctx, sb, sb,
                                        core="ro8" if a.tree > 0 else "seq"),
                             TW.replace(TW.P4(), sd_bw=a.sd))["time"] / max(tau, 1e-9)
            print(f"\n  [{m.group(1)} tokens | PC {float(t.group(1)):.2f} s/token | {tau:.2f} tokens/pass | "
                  f"P4 est. {p4:.1f} s/token at SD {a.sd:.0f} MB/s"
                  f"{', ro8 core' if a.tree > 0 else ''}]")
        history.append((user, reply))
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    main()
