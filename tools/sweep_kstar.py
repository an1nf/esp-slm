#!/usr/bin/env python3
"""Measure the lane knee k* = beta*F/B with tern lanes and check the R1 model."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import store_sim as ss
from tbin import FMT_T3, HEAD_I8, random_params, read_header, write_tbin

CFG = dict(dim=2560, n_layers=4, n_heads=20, n_kv=5, head_dim=128, inter=6912,
           vocab=8192, max_seq=16, rope_theta=500000.0, eps=1e-5)
NS = (1, 2, 4, 8, 16, 32, 64)
BWS = (400.0, 100.0, 25.0)
RING_E1 = 256
STEPS = 3
E2_BW = dict(psram=160.0 * 2, flash=30.0 * 2, sd=16.2 * 2)
E2_NS = (1, 8, 32)
E2_RINGS = (16, 256)
CHUNK = 16384


def run(binary, args):
    r = subprocess.run([binary, "lanes"] + args, capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    if r.returncode != 0 or line is None:
        raise SystemExit(f"failed: tern lanes {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    d = {}
    for kv in line.split()[1:]:
        k, v = kv.split("=")
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = v
    return d


def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def fit_knee(ns, ts, a_ref, b_ref):
    best = None
    for k in range(0, len(ns) + 1):
        flat, slope = ts[:k], list(zip(ns[k:], ts[k:]))
        c = statistics.mean(flat) if flat else 0.0
        if len(slope) >= 2:
            a2, b2 = fit_line([x for x, _ in slope], [y for _, y in slope])
        else:
            a2, b2 = a_ref, b_ref
        pred = [max(c, a2 + b2 * n) if flat else a2 + b2 * n for n in ns]
        sse = sum(((p - t) / t) ** 2 for p, t in zip(pred, ts))
        if best is None or sse < best[0]:
            knee = (c - a2) / b2 if flat else 0.0
            if k == len(ns):
                knee = math.inf
            best = (sse, c, a2, b2, knee, k)
    return best[1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", default="build/tern")
    ap.add_argument("--model", default="build/kstar_bitnet4l.tbin")
    ap.add_argument("--out", default="results/kstar")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(a.model):
        print("writing synthetic model")
        write_tbin(a.model, CFG, random_params(CFG, seed=21), FMT_T3, HEAD_I8)
    data = dict(machine=platform.processor() or platform.machine(),
                cpus=os.cpu_count(), ref={}, e1={}, e2=[])

    print("compute-only reference (sync, resident)")
    for n in NS:
        r = run(a.bin, [a.model, "--lanes", str(n), "--steps", str(STEPS), "--sync"])
        data["ref"][n] = r
        print(f"  N={n:3d}  {r['wall_ms']:9.1f} ms")
    T0 = [data["ref"][n]["wall_ms"] / 1e3 for n in NS]
    a0, b0 = fit_line(list(NS), T0)
    bytes_pass = data["ref"][1]["bytes_pass"]
    macs_lane = data["ref"][1]["macs_lane"]
    data["fit"] = dict(a=a0, b=b0, bytes_pass=bytes_pass, macs_lane=macs_lane)
    print(f"  a = {a0 * 1e3:.1f} ms, b = {b0 * 1e3:.2f} ms/lane, "
          f"{bytes_pass / 1e6:.1f} MB/pass")

    us = ss.units(read_header(a.model))
    place_all = os.path.join(os.path.dirname(a.model), "kstar_all_sd.place")
    ss.write_placement(place_all, {u.id: [("sd", 0, u.rows)] for u in us}, "E1")
    rows = []
    for B in BWS:
        tio = bytes_pass / (B * 1e6)
        nstar = (tio - a0) / b0
        ts = []
        for n in NS:
            r = run(a.bin, [a.model, "--lanes", str(n), "--steps", str(STEPS),
                            "--place", place_all, "--ring", str(RING_E1),
                            "--bw", f"sd={B}"])
            t = r["wall_ms"] / 1e3
            ts.append(t)
            tmax = max(tio, a0 + b0 * n)
            tsum = tio + a0 + b0 * n
            rows.append(dict(B=B, n=n, t=t, tmax=tmax, tsum=tsum, nstar=nstar,
                             wait=r["wait_ms"] / 1e3, mismatch=r["mismatch"]))
            print(f"  B={B:5.0f} N={n:3d}  measured {t * 1e3:8.1f}  max {tmax * 1e3:8.1f}"
                  f"  sum {tsum * 1e3:8.1f} ms  (n*={nstar:.1f})")
        data["e1"][B] = dict(nstar=nstar, tio=tio, ts=ts, knee=fit_knee(list(NS), ts, a0, b0))

    err_max = [abs(r["t"] - r["tmax"]) / r["t"] for r in rows]
    c1 = statistics.median(err_max) <= 0.10 and max(err_max) <= 0.25
    region = [r for r in rows if 0.5 * r["nstar"] <= r["n"] <= 2 * r["nstar"]]
    wins = [abs(r["t"] - r["tmax"]) < abs(r["t"] - r["tsum"]) for r in region]
    c2 = bool(region) and sum(wins) / len(wins) >= 0.8
    c3_detail = []
    for B in BWS:
        e = data["e1"][B]
        if e["nstar"] > 2:
            t1 = e["ts"][0]
            c3_detail.append((B, t1, e["tio"], abs(t1 - e["tio"]) / e["tio"] <= 0.10))
    c3 = all(x[3] for x in c3_detail)
    c4_detail = []
    for B in BWS:
        e = data["e1"][B]
        c, a2, b2, knee, k = e["knee"]
        ns_ = e["nstar"]
        if 1 <= ns_ <= NS[-1]:
            ok = (math.isfinite(knee) and (abs(knee - ns_) / ns_ <= 0.30
                                           or 0.5 <= knee / ns_ <= 2.0))
            c4_detail.append((B, ns_, knee, ok, "in grid"))
        else:
            ok = k == len(NS) or (math.isfinite(knee) and knee >= NS[-1] / 2)
            c4_detail.append((B, ns_, knee, ok, "off grid, flat ceiling required"))
    c4 = all(x[3] for x in c4_detail)

    hdr = read_header(a.model)
    total = sum(u.bytes for u in us)
    plat = ss.platform("R", hdr, 1)
    for t in ss.TIERS:
        plat.chans[t] = ss.Channel(E2_BW[t] * 1e6)
    plat.caps["psram"], plat.caps["flash"] = 0.10 * total, 0.20 * total
    tgt = ss.targets(total, plat)
    trace = [r for r in ss.analytic_trace(hdr, 1, ss.KV_F32) if r[1] in ("W", "H")]
    e2rows = []
    for pol in ("stripe", "fill"):
        plan = ss.place(us, tgt, pol)
        pfile = os.path.join(os.path.dirname(a.model), f"kstar_e2_{pol}.place")
        ss.write_placement(pfile, plan, f"E2 {pol}")
        base_steps = ss.steps_of(trace, us, plan, CHUNK)
        for ring in E2_RINGS:
            for n in E2_NS:
                steps = [(t, b, a0 * b / bytes_pass + b0 * n * m / macs_lane)
                         for t, b, m in base_steps]
                slot = 16 * ((max(s[1] for s in base_steps) + 15) // 16)
                depth = {t: max(1, (ring * 1024) // slot) * slot for t in ss.TIERS}
                p2 = ss.Platform("E2", {t: ss.Channel(E2_BW[t] * 1e6, 0.0, depth[t])
                                        for t in ss.TIERS}, plat.caps, 1.0)
                tsim = ss.simulate(steps, p2)
                r = run(a.bin, [a.model, "--lanes", str(n), "--steps", str(STEPS),
                                "--place", pfile, "--ring", str(ring),
                                "--bw", ",".join(f"{t}={E2_BW[t]}" for t in ss.TIERS)])
                t = r["wall_ms"] / 1e3
                e2rows.append(dict(policy=pol, ring=ring, n=n, t=t, tsim=tsim,
                                   err=abs(t - tsim) / tsim, mismatch=r["mismatch"]))
                print(f"  E2 {pol:6s} ring {ring:3d} KiB N={n:2d}  measured "
                      f"{t * 1e3:8.1f}  simulated {tsim * 1e3:8.1f} ms")
    data["e2"] = e2rows
    c5 = statistics.median(r["err"] for r in e2rows) <= 0.15
    c6_detail = []
    for ring in E2_RINGS:
        for n in E2_NS:
            s_ = next(r for r in e2rows if r["policy"] == "stripe" and r["ring"] == ring and r["n"] == n)
            f_ = next(r for r in e2rows if r["policy"] == "fill" and r["ring"] == ring and r["n"] == n)
            if abs(f_["tsim"] - s_["tsim"]) / s_["tsim"] >= 0.10:
                c6_detail.append((ring, n, (f_["tsim"] > s_["tsim"]) == (f_["t"] > s_["t"])))
    c6 = all(x[2] for x in c6_detail) if c6_detail else None
    mism = any(r["mismatch"] for r in rows) or any(r["mismatch"] for r in e2rows)

    def ok(x):
        return "n/a" if x is None else ("**PASS**" if x else "**FAIL**")
    L = ["# k* sweep\n",
         f"Machine: {data['machine']}, {data['cpus']} CPUs. Synthetic model with BitNet 2B4T "
         f"shapes, 4 layers, vocab 8192: {bytes_pass / 1e6:.1f} MB and "
         f"{macs_lane / 1e6:.0f} M MAC per pass and lane. Median of {STEPS} passes.\n",
         "## Compute-only reference\n",
         f"T0(N) = a + b*N with a = {a0 * 1e3:.1f} ms and b = {b0 * 1e3:.2f} ms per lane "
         f"(~{macs_lane / b0 / 1e9:.1f} GMAC/s per lane).\n",
         "| N | " + " | ".join(str(n) for n in NS) + " |",
         "|---|" + "---|" * len(NS),
         "| T0 (ms) | " + " | ".join(f"{t * 1e3:.0f}" for t in T0) + " |\n",
         "## E1: one tier (I/O ceiling = bytes/B, 256 KiB ring)\n",
         "| B (MB/s) | predicted n* | " + " | ".join(f"N={n}" for n in NS) + " |",
         "|---|---|" + "---|" * len(NS)]
    for B in BWS:
        e = data["e1"][B]
        cells = []
        for n, t in zip(NS, e["ts"]):
            tmax = max(e["tio"], a0 + b0 * n)
            cells.append(f"{t * 1e3:.0f} / {tmax * 1e3:.0f}")
        L.append(f"| {B:.0f} | {e['nstar']:.1f} | " + " | ".join(cells) + " |")
    L.append("\n(measured / predicted by max(bytes/B, a + b*N), ms)\n")
    L.append(f"1. Global fit: median error {100 * statistics.median(err_max):.1f} %, "
             f"max {100 * max(err_max):.1f} % -> {ok(c1)}")
    L.append(f"2. Knee region: max beats sum at "
             f"{sum(wins)}/{len(wins)} points -> {ok(c2)}")
    L.append("3. I/O ceiling at N = 1: " + "; ".join(
        f"B={B:.0f}: {t1 * 1e3:.0f} ms vs {tio * 1e3:.0f} ms" for B, t1, tio, _ in c3_detail)
        + f" -> {ok(c3)}")
    L.append("4. Knee: " + "; ".join(
        f"B={B:.0f}: predicted {ns_:.1f}, fitted "
        f"{'inf' if not math.isfinite(kn) else f'{kn:.1f}'} ({why})"
        for B, ns_, kn, _, why in c4_detail) + f" -> {ok(c4)}\n")
    L.append("## E2: three tiers (PSRAM : flash : SD = 320 : 60 : 32.4 MB/s; capacities "
             "10 % and 20 %)\n")
    L.append("| Placement | Ring | N | Measured (ms) | Simulated (ms) | Error |")
    L.append("|---|---|---|---|---|---|")
    for r in e2rows:
        L.append(f"| {r['policy']} | {r['ring']} KiB | {r['n']} | {r['t'] * 1e3:.0f} | "
                 f"{r['tsim'] * 1e3:.0f} | {100 * r['err']:.1f} % |")
    L.append(f"\n5. Median measured vs simulated error: "
             f"{100 * statistics.median(r['err'] for r in e2rows):.1f} % -> {ok(c5)}")
    L.append(f"6. stripe/fill order where the simulator predicts >= 10 %: "
             f"{sum(x[2] for x in c6_detail)}/{len(c6_detail)} -> {ok(c6)}\n")
    L.append(f"Delivered chunks differing from requested: {'yes (ERROR)' if mism else 'none'}.\n")
    verdict = all(x for x in (c1, c2, c3, c4, c5) if x is not None) and (c6 is not False)
    L.append(f"**Verdict: {'all criteria pass' if verdict else 'at least one criterion fails'}.**")
    data["criteria"] = dict(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, verdict=verdict)
    with open(os.path.join(a.out, "kstar.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.out, "kstar.json"), "w") as f:
        json.dump(data, f, indent=1, default=str)
    print("\n".join(L))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
