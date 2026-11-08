"""State/KV quantization identities, tree exactness and datastore drafts."""
import os
import subprocess
import sys
import tempfile

q35, out = sys.argv[1], sys.argv[2]
tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
ids = "3,1,4,1,5,9,2,6"
seq200 = ",".join(str((37 * i + 11) % 500) for i in range(200))
fails = 0


def report(name, ok):
    global fails
    print(f"  {name}: {'OK' if ok else 'FAIL'}")
    fails += not ok


def kl_table(model, cands):
    r = subprocess.run([q35, "cmpstate", model, "--cands", cands, "--ids", seq200],
                       capture_output=True, text=True)
    rows = [l for l in r.stdout.splitlines() if l.startswith("| ") and not l.startswith("| candidate")]
    return r.returncode, {l.split("|")[1].strip().split(",")[0]: float(l.split("|")[3]) for l in rows}


rc, kl = kl_table(f"{out}/q35_d.gguf", "f16,int8+h100,ro16:4:2,ro8+h100:4:2,f32+k4,f32+k4+r16,"
                                       "f32+k2,f32+k2+r16,f32+k2+r999,int8,int8+d+s,ro8+d+s+x16:4:2")
report("cmpstate runs all candidates", rc == 0 and len(kl) == 12)
report("int8 with all heads 16-bit = f16", kl.get("int8+h100") == kl.get("f16"))
report("ro8 with all heads 16-bit = ro16", kl.get("ro8+h100 rank 4") == kl.get("ro16 rank 4"))
report("kv2 refine >= ctx is exact", kl.get("f32+k2+r999") == 0.0)
report("kv refine lowers KL (4 and 2 bits)",
       kl.get("f32+k4+r16", 1) < kl.get("f32+k4", 0) and kl.get("f32+k2+r16", 1) < kl.get("f32+k2", 0))

for m in ("q35_a.gguf", "q35_b.gguf", "q35_d.gguf", "q35_e.gguf"):
    for extra in (("--state", "int8", "--dither", "1", "--smooth", "1", "--hi", "0.3"),
                  ("--kv-bits", "4", "--kv-refine", "4"),
                  ("--kv-bits", "2", "--kv-refine", "8", "--kv-recent", "4", "--temp", "0.8", "--top-k", "20")):
        r = subprocess.run([q35, "spec", f"{out}/{m}", "--ids", ids, "--gen", "40", "--ctx", "80",
                            "--tree", "12", "--gamma", "6", *extra, "--check"], capture_output=True, text=True)
        report(f"tree = sequential, {m} {' '.join(extra)}", r.returncode == 0 and "CHECK OK" in r.stdout)

r = subprocess.run([q35, "treetest", f"{out}/q35_d.gguf", "--ids", ids, "--gamma", "12", "--seed", "3",
                    "--state", "ro8", "--dither", "1", "--smooth", "1", "--x16", "1", "--ro-rank", "4",
                    "--tol", "0.5"], capture_output=True, text=True)
report("ro8 dither+smooth+x16 tree vs f32", r.returncode == 0 and r.stdout.strip().endswith("OK"))


def passes(model, *extra):
    r = subprocess.run([q35, "spec", model, "--ids", ids, "--gen", "40", "--ctx", "80", "--gamma", "6",
                        *extra], capture_output=True, text=True)
    line = next(l for l in r.stdout.splitlines() if l.startswith("spec:"))
    gen = next(l for l in r.stdout.splitlines() if l.startswith("ids: "))[5:]
    return r.returncode, int(line.split(" passes")[0].split()[-1]), gen, r.stdout


d = tempfile.mkdtemp()
hds = os.path.join(d, "t.hds")
for m in ("q35_a.gguf", "q35_b.gguf"):
    _, _, gen, _ = passes(f"{out}/{m}", "--drafter", "none")
    subprocess.run([sys.executable, os.path.join(tools, "datastore.py"), hds, "--ids", f"{ids},{gen}",
                    "--ids", "7,7,7,1,2,3"], check=True, capture_output=True)
    rc0, p0, _, _ = passes(f"{out}/{m}", "--tree", "12")
    rc1, p1, _, o1 = passes(f"{out}/{m}", "--tree", "12", "--ds", hds, "--check")
    report(f"datastore drafts {m}: {p0} -> {p1} passes", rc0 == 0 and rc1 == 0 and p1 < p0
           and "CHECK OK" in o1)
os.remove(hds)
print("quantization and datastore:", "FAIL" if fails else "OK")
sys.exit(1 if fails else 0)
