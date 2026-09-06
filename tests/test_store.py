#!/usr/bin/env python3
"""Tiered store: placement, trace and counters."""
from __future__ import annotations

import argparse
import filecmp
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import store_sim as ss
from ref_np import KV_F32, KV_I8, RefModel, greedy_reference, write_ref
from tbin import FMT_T3, HEAD_I8, random_params, read_header, write_tbin

BITNET_2L = dict(dim=2560, n_layers=2, n_heads=20, n_kv=5, head_dim=128, inter=6912,
                 vocab=4096, max_seq=64, rope_theta=500000.0, eps=1e-5)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def make_placement(tbin: str, policy: str, path: str) -> dict:
    cfg = read_header(tbin)
    us = ss.units(cfg)
    total = sum(u.bytes for u in us)
    plat = ss.platform("R", cfg, 16)
    plat.caps["psram"], plat.caps["flash"] = 0.3 * total, 0.2 * total
    plan = ss.place(us, ss.targets(total, plat), policy)
    ss.write_placement(path, plan, f"test {policy}")
    return ss.placed_bytes(us, plan)


def case(binary: str, out: str, tbin: str, ref: str, kv: str, policy: str,
         buf_kib: int, tag: str, chunk_kib: int = 1) -> bool:
    place = os.path.join(out, f"{tag}.place")
    placed = make_placement(tbin, policy, place)
    res = os.path.join(out, f"{tag}_res.bin")
    got = os.path.join(out, f"{tag}_placed.bin")
    trace = os.path.join(out, f"{tag}.trace")
    rc1, o1 = run([binary, "test", tbin, ref, "--kv", kv, "--logits", res])
    rc2, o2 = run([binary, "test", tbin, ref, "--kv", kv, "--logits", got,
                   "--place", place, "--buf", str(buf_kib), "--chunk", str(chunk_kib),
                   "--trace", trace])
    parity = rc1 == 0 and rc2 == 0
    same = os.path.exists(res) and os.path.exists(got) \
        and filecmp.cmp(res, got, shallow=False)
    m = re.search(r"\(psram,flash,sd\): (\d+),(\d+),(\d+)", o2)
    counters = [int(x) for x in m.groups()] if m else None

    cfg = read_header(tbin)
    tr = ss.read_trace(trace)
    kvf = KV_I8 if kv == "int8" else KV_F32
    trace_ok = True
    for p in sorted({r[0] for r in tr}):
        rows = [r for r in tr if r[0] == p]
        trace_ok &= rows == ss.analytic_trace(cfg, p, kvf, rows[0][3])
    plan = ss.read_placement(place)
    per = {t: 0 for t in ss.TIERS}
    for t, b, _m in ss.steps_of([r for r in tr if r[1] in ("E", "W", "H")],
                                ss.units(cfg), plan, 1 << 62):
        per[t] += b
    m2 = re.search(r"reads per tier \(psram,flash,sd\): (\d+),(\d+),(\d+)", o2)
    reads = [int(x) for x in m2.groups()] if m2 else None
    nread = {t: 0 for t in ss.TIERS}
    for t, _b, _m in ss.steps_of([r for r in tr if r[1] in ("E", "W", "H")],
                                 ss.units(cfg), plan, chunk_kib * 1024):
        nread[t] += 1
    cnt_ok = counters == [per[t] for t in ss.TIERS] \
        and reads == [nread[t] for t in ss.TIERS]
    used = all(placed[t] > 0 for t in ss.TIERS)
    ok = parity and same and trace_ok and cnt_ok and used
    first = o2.splitlines()[0] if o2 else ""
    print(f"  {tag:<28} {first}")
    print(f"  {'':<28} placed logits = resident: {'yes' if same else 'NO'}; "
          f"trace = analytic: {'yes' if trace_ok else 'NO'}; bytes {counters} and "
          f"reads {reads} per tier = model: {'yes' if cnt_ok else 'NO'}; "
          f"all three tiers used: {'yes' if used else 'NO'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="build")
    ap.add_argument("--bin", default="build/tern")
    ap.add_argument("--skip-big", action="store_true")
    a = ap.parse_args()
    b, o = a.bin, a.out
    ok = True
    ok &= case(b, o, f"{o}/tiny_t3_h8.tbin", f"{o}/tiny_h8kv8.ref", "int8", "stripe", 1,
               "tiny_t3_h8_kv8_stripe")
    ok &= case(b, o, f"{o}/tiny_t3_h8.tbin", f"{o}/tiny_h8kv8.ref", "int8", "fill", 1,
               "tiny_t3_h8_kv8_fill")
    ok &= case(b, o, f"{o}/tiny_t2.tbin", f"{o}/tiny.ref", "f32", "stripe", 1,
               "tiny_t2_h32_kv32_stripe")

    if not a.skip_big:
        tb = os.path.join(o, "bitnet2l_t3_h8.tbin")
        ref = os.path.join(o, "bitnet2l_h8kv8.ref")
        if not (os.path.exists(tb) and os.path.exists(ref)):
            print("  writing a 2-layer model with BitNet 2B4T shapes")
            p = random_params(BITNET_2L, seed=11)
            write_tbin(tb, BITNET_2L, p, FMT_T3, HEAD_I8)
            rng = np.random.default_rng(12)
            prompt = rng.integers(0, BITNET_2L["vocab"], size=3).tolist()
            g, lg = greedy_reference(RefModel(BITNET_2L, p, HEAD_I8, KV_I8), prompt, 4)
            write_ref(ref, BITNET_2L["vocab"], prompt, g, lg)
            del p
        ok &= case(b, o, tb, ref, "int8", "stripe", 64, "bitnet2l_stripe", 16)
        ok &= case(b, o, tb, ref, "int8", "fill", 16, "bitnet2l_fill", 8)
    print("TIERED STORE:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
