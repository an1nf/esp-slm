"""Speculative decoding: exactness, sampling, tree and adjacency."""
import os
import subprocess
import sys

q35, out = sys.argv[1], sys.argv[2]
ids = "3,1,4,1,5,9,2,6"


def spec(model, *extra):
    r = subprocess.run([q35, "spec", model, "--ids", ids, "--gen", "40", "--ctx", "80",
                        "--gamma", "6", *extra], capture_output=True, text=True)
    return r.returncode, r.stdout


fails = 0
for m in ("q35_a.gguf", "q35_b.gguf"):
    model = f"{out}/{m}"
    for st in ("f32", "int8"):
        rc, o = spec(model, "--state", st, "--drafter", "none")
        gen = next(l for l in o.splitlines() if l.startswith("ids: "))[5:]
        for c in ("0", "0.15", "0.5"):
            rc, o = spec(model, "--state", st, "--oracle", gen, "--corrupt", c, "--check")
            line = next(l for l in o.splitlines() if l.startswith("spec:"))
            ok = rc == 0 and "CHECK OK" in o
            if st == "f32" and c == "0":
                ok = ok and "accepted 34 (1.00)" in line
            print(f"  {m} state {st:4s} errors {c:4s}: {'OK  ' if ok else 'FAIL'} {line[6:]}")
            fails += not ok
        for dr in ("pld", "jacobi", "mix"):
            rc, o = spec(model, "--state", st, "--drafter", dr, "--check")
            ok = rc == 0 and "CHECK OK" in o
            print(f"  {m} state {st:4s} drafter {dr:6s}: {'OK' if ok else 'FAIL'}")
            fails += not ok
        passes = {}
        for kind, extra in (("chain", ()), ("tree", ("--tree", "12"))):
            rc, o = spec(model, "--state", st, "--oracle", gen, "--corrupt", "0", *extra)
            line = next(l for l in o.splitlines() if l.startswith("spec:"))
            passes[kind] = int(line.split(" passes")[0].split()[-1])
        for c in ("0", "0.15", "0.5"):
            rc, o = spec(model, "--state", st, "--oracle", gen, "--corrupt", c, "--tree", "12", "--check")
            ok = rc == 0 and "CHECK OK" in o
            if st == "f32" and c == "0":
                ok = ok and passes["tree"] == passes["chain"]
            print(f"  {m} state {st:4s} tree 12, errors {c:4s}: {'OK' if ok else 'FAIL'}")
            fails += not ok
        for nt in ("4", "16", "32"):
            rc, o = spec(model, "--state", st, "--tree", nt, "--check")
            ok = rc == 0 and "CHECK OK" in o
            print(f"  {m} state {st:4s} tree {nt:2s} no oracle: {'OK' if ok else 'FAIL'}")
            fails += not ok
for m in ("q35_a.gguf", "q35_b.gguf"):
    for extra in (("--drafter", "mix"), ("--tree", "12"), ("--tree", "24", "--presence", "1.5")):
        for seed in ("1", "7"):
            rc, o = spec(f"{out}/{m}", *extra, "--temp", "0.9", "--top-k", "20", "--top-p", "0.95",
                         "--seed", seed, "--check")
            ok = rc == 0 and "CHECK OK" in o
            print(f"  sampling {m} {' '.join(extra):28s} seed {seed}: {'OK' if ok else 'FAIL'}")
            fails += not ok
for m in ("q35_a.gguf", "q35_b.gguf", "q35_d.gguf", "q35_e.gguf"):
    r = subprocess.run([q35, "treetest", f"{out}/{m}", "--ids", ids, "--gamma", "12", "--seed", "3",
                        "--state", "ro32", "--ro-rank", "4", "--tol", "0.25"], capture_output=True, text=True)
    ok = r.returncode == 0 and r.stdout.strip().endswith("OK")
    print(f"  read-only ro32 (tree, rank 4) {m}: {'OK' if ok else 'FAIL'}")
    fails += not ok
seq200 = ",".join(str((37 * i + 11) % 500) for i in range(200))
r = subprocess.run([q35, "cmpstate", f"{out}/q35_d.gguf", "--cands", "ro32:4:2,ro32:16:16,ro16:16:2,ro8:16:2",
                    "--ids", seq200], capture_output=True, text=True)
rows = [l for l in r.stdout.splitlines() if l.startswith("| ro")]
kls = [float(l.split("|")[3]) for l in rows]
ok = r.returncode == 0 and len(rows) == 4 and max(kls[:2]) <= 1e-4
print(f"  cmpstate (ro32: KL {kls[0] if kls else float('nan'):.1e}/{kls[1] if len(kls) > 1 else float('nan'):.1e} <= 1e-4; "
      f"ro16/ro8 run): {'OK' if ok else 'FAIL'}")
fails += not ok
r = subprocess.run([q35, "cmpstate", f"{out}/q35_d.gguf", "--cands", "f32:16:2,f32n:8:16,f32n:5:16",
                    "--ids", seq200], capture_output=True, text=True)
kl3 = [float(l.split("|")[3]) for l in r.stdout.splitlines() if l.startswith("| f32")]
ok = r.returncode == 0 and len(kl3) == 3 and kl3[0] == 0.0 and kl3[1] == 0.0 and kl3[2] > 0.0
print(f"  cmpstate f32 control (chunk 2 and noise 1e-8 = 0; noise 1e-5 > 0): {'OK' if ok else 'FAIL'}")
fails += not ok
import tempfile
adjf = os.path.join(tempfile.mkdtemp(), "t.adj")
for m in ("q35_a.gguf", "q35_b.gguf"):
    for extra in (("--tree", "16", "--adj", "1", "--adj-file", adjf),
                  ("--tree", "32", "--adj", "1", "--adj-file", adjf, "--temp", "0.8", "--top-k", "20"),
                  ("--tree", "16", "--adj", "1", "--adj-file", adjf, "--adj-ro", "1")):
        rc, o = spec(f"{out}/{m}", *extra, "--check")
        ok = rc == 0 and "CHECK OK" in o
        print(f"  adjacency {m} {' '.join(x for x in extra if x != adjf):40s}: {'OK' if ok else 'FAIL'}")
        fails += not ok
    if os.path.exists(adjf):
        os.remove(adjf)
for m in ("q35_a.gguf", "q35_b.gguf", "q35_d.gguf", "q35_e.gguf"):
    for st, kv in (("f32", "f16"), ("int8", "f32")):
        r = subprocess.run([q35, "treetest", f"{out}/{m}", "--ids", ids, "--gamma", "16", "--seed", "5",
                            "--state", st, "--kv", kv], capture_output=True, text=True)
        ok = r.returncode == 0 and "OK" in r.stdout
        print(f"  treetest {m} state {st:4s} kv {kv}: {'OK' if ok else 'FAIL'}")
        fails += not ok
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import q35_tiny
rc, o = spec(f"{out}/q35_a.gguf", "--drafter", "none")
gen = [int(t) for t in next(l for l in o.splitlines() if l.startswith("ids: "))[5:].split(",")]
eos = gen[5]
first = gen.index(eos)
q35_tiny.build(f"{out}/q35_a_eos.gguf", 1, 4, fused=False, untied_q8=False, f16_gate=False,
               sect=[3, 3, 2, 0], scale_up=False, eos=eos)
for dr in ("none", "mix", "tree"):
    extra = ("--tree", "12") if dr == "tree" else ("--drafter", dr)
    rc, o = spec(f"{out}/q35_a_eos.gguf", *extra, "--stop", "--check")
    got = [int(t) for t in next(l for l in o.splitlines() if l.startswith("ids: "))[5:].split(",")]
    ok = rc == 0 and got == gen[:first + 1]
    print(f"  stop at EOS ({dr}): {'OK' if ok else 'FAIL'} ({len(got)} tokens, expected {first + 1})")
    fails += not ok
print("draft verification:", "FAIL" if fails else "OK")
sys.exit(1 if fails else 0)
