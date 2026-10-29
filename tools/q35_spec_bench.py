"""Weight-free drafters on real prompts, with P4 time estimates."""
import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q35_cmp as C
import p4_twin as TW

CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

PROMPTS = {
    "summary": CHAT.format(
        "Summarize the following text in three bullet points:\n\n"
        "The ESP32-P4 is an Espressif microcontroller with two RISC-V cores at 400 MHz, "
        "custom vector instructions for AI and signal processing, and up to 32 MB of PSRAM "
        "in the same package. It has no Wi-Fi or Bluetooth: it targets products with a display "
        "and camera, such as control panels, video doorbells or payment terminals, and is paired "
        "with another chip when it needs connectivity. It supports SD cards in UHS-I mode, "
        "high-speed USB 2.0 and Ethernet. Its main limit for running large language models is "
        "memory: the weights of a two-billion-parameter model do not fit in PSRAM and must be "
        "read from the SD card for every token, so card speed almost fully decides generation "
        "speed."),
    "code": CHAT.format(
        "Add type hints and a docstring to this Python function without changing its logic, "
        "and return the full function:\n\n"
        "def moving_average(values, window):\n"
        "    if window <= 0:\n"
        "        raise ValueError('window must be positive')\n"
        "    out = []\n"
        "    acc = 0.0\n"
        "    for i, v in enumerate(values):\n"
        "        acc += v\n"
        "        if i >= window:\n"
        "            acc -= values[i - window]\n"
        "        if i >= window - 1:\n"
        "            out.append(acc / window)\n"
        "    return out\n"),
    "question": CHAT.format("Why is the sky blue? Explain it for a ten-year-old."),
    "list": CHAT.format("List the countries of South America with their capitals, one per line, "
                        "in the format Country: Capital."),
}

RUNS = [("none", 0, False), ("pld", 8, False), ("mix", 8, True), ("mix", 16, False)]


def parse(out):
    m = re.search(r"spec: (\d+) tokens in (\d+) passes \(([\d.]+) tokens/pass\).*mean T ([\d.]+)", out)
    t = re.search(r"decode ([\d.]+) s", out)
    src = re.search(r"by source: (.*)", out)
    return dict(tokens=int(m.group(1)), passes=int(m.group(2)), tau=float(m.group(3)),
                T=float(m.group(4)), secs=float(t.group(1)), src=src.group(1) if src else "",
                check=("CHECK OK" in out) if "CHECK" in out else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--q35", default="build/q35")
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0]
    log = open(os.path.join(a.out, f"{tag}.spec.log"), "w")
    info = subprocess.run([a.q35, "info", a.model], capture_output=True, text=True).stdout
    E = int(re.search(r"embd (\d+)", info).group(1))
    size = {2048: "2B", 2560: "4B", 4096: "9B"}.get(E, "2B")
    rows, texts = [], {}
    for name, text in PROMPTS.items():
        tok = os.path.join(a.out, f"{tag}.{name}.ids.bin")
        r = subprocess.run([a.llama, a.model, tok, "--text", text, "--tokens-only", "1"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        ids, _ = C.load(tok)
        ids_s = ",".join(str(int(t)) for t in ids)
        for dr, gamma, chk in RUNS:
            cmd = [a.q35, "spec", a.model, "--ids", ids_s, "--gen", str(a.gen),
                   "--ctx", str(len(ids) + a.gen + 40), "--drafter", dr,
                   "--gamma", str(max(gamma, 1))]
            if chk:
                cmd += ["--check", "--print"]
            print(f"{name}: {dr} gamma={gamma}", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            log.write("$ " + " ".join(cmd[:4]) + f" {dr} {gamma}\n{r.stdout}\n{r.stderr[-1500:]}\n")
            log.flush()
            if r.returncode:
                sys.exit(f"q35 spec failed ({name}, {dr}):\n{r.stdout}\n{r.stderr[-1500:]}")
            p = parse(r.stdout)
            p.update(name=name, dr=dr, gamma=gamma, wall=time.time() - t0)
            if chk:
                txt = r.stdout[r.stdout.find("\n", r.stdout.find("ids: ")) + 1:]
                texts[name] = txt.split("CHECK")[0].strip()
            for sd in (40, 80):
                pas = TW.run_pass(TW.Variant(size, max(1, round(p["T"])), 1024,
                                             2 if size == "2B" else 1, 2 if size == "2B" else 1),
                                  TW.replace(TW.P4(), sd_bw=sd))["time"]
                p[f"p4_{sd}"] = pas / p["tau"]
            rows.append(p)
    L = [f"# Weight-free drafters, {tag} ({size})", "",
         f"{a.gen} tokens per prompt, greedy. tau = tokens per pass; T = tokens evaluated per "
         f"pass. P4 s/token from p4_twin (state and KV {'f16' if size == '2B' else 'int8'}, ctx 1024).", "",
         "| Prompt | Drafter | gamma | tau | mean T | Sources (accepted/proposed) | P4 SD 40 (s/token) | P4 SD 80 (s/token) | check |",
         "|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        chk = "" if p["check"] is None else ("OK" if p["check"] else "FAIL")
        L.append(f"| {p['name']} | {p['dr']} | {p['gamma']} | {p['tau']:.2f} | {p['T']:.1f} | "
                 f"{p['src']} | {p['p4_40']:.1f} | {p['p4_80']:.1f} | {chk} |")
    L += ["", "## Generated text (mix, gamma = 8)", ""]
    for k, v in texts.items():
        L += [f"### {k}", "", "```", v, "```", ""]
    rep = "\n".join(L)
    with open(os.path.join(a.out, f"{tag}.spec.md"), "w") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
