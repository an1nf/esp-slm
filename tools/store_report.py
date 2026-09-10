#!/usr/bin/env python3
"""Tiered store report for BitNet b1.58 2B4T on the ESP32-P4."""
from __future__ import annotations

import math

import bytes_sim as bm
import store_sim as ss

KiB = 1024
MB = 1e6
CTX = 512
CHUNK = 16 * KiB
BUFS = (0, 8 * KiB, 16 * KiB, 32 * KiB, 64 * KiB, 128 * KiB, 256 * KiB, math.inf)


def fmt_t(s: float) -> str:
    return f"{s:.2f} s" if s >= 1 else f"{s * 1000:.0f} ms"


def fmt_buf(d: float) -> str:
    if d == 0:
        return "0 (inline)"
    if d == math.inf:
        return "inf"
    return f"{int(d // KiB)} KiB"


def setup(scenario: str, policy: str, cfg: dict, rate: float = bm.COMPUTE_MACS,
          sd_lat: float = 0.0):
    us = ss.units(cfg)
    plat = ss.platform(scenario, cfg, CTX, ss.KV_I8, rate=rate, sd_lat=sd_lat)
    total = sum(u.bytes for u in us)
    fixed = (CTX * ss.kv_bytes_per_pos(cfg, ss.KV_I8) + ss.aux_bytes(cfg)) \
        / plat.chans["psram"].bw
    tgt = ss.targets(total, plat, fixed)
    plan = ss.place(us, tgt, policy)
    trace = ss.analytic_trace(cfg, CTX - 1, ss.KV_I8, token=cfg["vocab"] // 2)
    return us, plat, plan, trace, tgt


def run(us, plat, plan, trace, depth: dict, chunk: int = CHUNK, macs_mul: float = 1.0,
        order: str = "interleave"):
    finite = [d for d in depth.values() if 0 < d < math.inf]
    ch = min([chunk] + [int(d // 2) for d in finite])
    steps = ss.steps_of(trace, us, plan, max(ch, 1), order)
    if macs_mul != 1.0:
        steps = [(t, b, m * macs_mul) for t, b, m in steps]
    return ss.simulate(steps, ss.with_depth(plat, depth)), ss.bounds(steps, plat)


def report() -> str:
    cfg = dict(ss.BITNET)
    out = ["# Tiered store: BitNet b1.58 2B4T on the ESP32-P4\n",
           f"Decode pass at position {CTX - 1} (ctx {CTX}), T3 body (1.6 bits/weight), int8 head "
           "with per-row scale, int8 KV cache. "
           f"Compute {bm.COMPUTE_MACS / 1e9:.0f} GMAC/s unless noted [H]. Read chunks of 16 KiB "
           "(or half the buffer, if smaller). Capacities as in bytes_sim: 32 MB PSRAM minus 2 MB "
           "reserve, KV and norms; 28 MB flash for weights.\n"]

    tr = ss.analytic_trace(cfg, CTX - 1, ss.KV_I8)
    cat = {}
    for _p, k, _u, _a, _b, nb, _m in tr:
        cat[k] = cat.get(k, 0) + nb
    v_int8 = next(v for v in bm.VARIANTS if v.label == "BitNet 2B, int8 head")
    pc = bm.pass_cost(v_int8, CTX)
    kv_bm = CTX * bm.BITNET_2B4T.kv_bytes_per_pos(1.0)
    out.append("## 1. Bytes per token: engine trace vs analytic model\n")
    out.append("| Item | tern (trace) | bytes_sim | Note |")
    out.append("|---|---|---|---|")
    rows = [("Ternary body", cat["W"], pc.body_bytes,
             "T3 = 1.6 bits/weight plus row padding; bytes_sim assumes TQ1_0 (1.6875)"),
            ("int8 head", cat["H"], pc.head_bytes, "f32 scale per row (4 B of 2564)"),
            ("KV cache (read)", cat["K"], kv_bm, "f32 scales per position and head"),
            ("f32 norms", cat["A"], 0, "not counted in bytes_sim"),
            ("Embedding row", cat["E"], 0, "one row per token")]
    for name, a, b, why in rows:
        out.append(f"| {name} | {a / MB:.2f} MB | {b / MB:.2f} MB | {why} |")
    tot = sum(cat.values())
    out.append(f"| **Total** | **{tot / MB:.2f} MB** | "
               f"**{(pc.body_bytes + pc.head_bytes + kv_bm) / MB:.2f} MB** | |\n")

    out.append("## 2. Time per token vs prefetch buffer\n")
    out.append("Per-channel SRAM buffer. *Inline* = cached reads without prefetch: no I/O-compute "
               "overlap. *stripe* splits each matrix across the three tiers in the optimal "
               "ratio; *interleave* walks its rows alternating tiers per chunk (what tern does), "
               "*sequential* reads the PSRAM part, then flash, then SD. *fill* fills PSRAM and "
               "flash with the first layers and leaves the rest on SD. The bound is "
               "max(parallel I/O, compute); serial is the sum of everything. No USB channel.\n")
    cols = (("stripe", "interleave", "sd", "stripe interleaved, DMA on SD only"),
            ("stripe", "interleave", "all", "stripe interleaved, DMA on all"),
            ("stripe", "sequential", "all", "stripe sequential, DMA on all"),
            ("fill", "interleave", "all", "fill, DMA on all"))
    for scen in ("R", "R+SD", "T"):
        res = {}
        for pol in ("stripe", "fill"):
            us, plat, plan, trace, tgt = setup(scen, pol, cfg)
            for p2, order, mode, _lbl in cols:
                if p2 != pol:
                    continue
                for d in BUFS:
                    depth = dict(psram=0, flash=0, sd=d) if mode == "sd" \
                        else dict(psram=d, flash=d, sd=d)
                    t, b = run(us, plat, plan, trace, depth, order=order)
                    res[(pol, order, mode, d)] = t
            if pol == "stripe":
                res["bounds"], res["tgt"] = b, tgt
        b, tgt = res["bounds"], res["tgt"]
        fast = (tgt["psram"] + tgt["flash"]) / sum(tgt.values())
        out.append(f"### Scenario {scen}\n")
        out.append(f"Optimal weight split: PSRAM {tgt['psram'] / MB:.1f} MB, flash "
                   f"{tgt['flash'] / MB:.1f} MB, SD {tgt['sd'] / MB:.1f} MB "
                   f"({100 * fast:.0f} % in fast memory). Parallel bound: "
                   f"**{fmt_t(b['parallel'])}**; serial: {fmt_t(b['serial'])}; compute "
                   f"only: {fmt_t(b['comp'])}.\n")
        out.append("| Buffer per channel | " + " | ".join(c[3] for c in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for d in BUFS:
            cells = []
            for pol, order, mode, _lbl in cols:
                t = res[(pol, order, mode, d)]
                cells.append(f"{fmt_t(t)} ({100 * b['parallel'] / t:.0f} %)")
            out.append(f"| {fmt_buf(d)} | " + " | ".join(cells) + " |")
        out.append("\n(in parentheses: fraction of the parallel bound reached)\n")

    out.append("## 3. Compute sensitivity (R+SD, stripe, 64 KiB on all)\n")
    out.append("| Effective compute | Time per token | Bound | Compute only |")
    out.append("|---|---|---|---|")
    for rate in (0.3e9, 1e9, 3e9, 10e9):
        us, plat, plan, trace, _ = setup("R+SD", "stripe", cfg, rate=rate)
        t, b = run(us, plat, plan, trace, dict(psram=64 * KiB, flash=64 * KiB, sd=64 * KiB))
        out.append(f"| {rate / 1e9:g} GMAC/s | {fmt_t(t)} | {fmt_t(b['parallel'])} | "
                   f"{fmt_t(b['comp'])} |")
    out.append("")

    out.append("## 4. SD read latency and chunk size (R, stripe)\n")
    out.append("SD buffer = 2 chunks; PSRAM and flash 64 KiB. Latency per read command [H].\n")
    out.append("| Chunk | latency 0 | 0.2 ms | 1 ms |")
    out.append("|---|---|---|---|")
    for chunk in (4 * KiB, 16 * KiB, 64 * KiB, 256 * KiB):
        cells = []
        for lat in (0.0, 0.2e-3, 1e-3):
            us, plat, plan, trace, _ = setup("R", "stripe", cfg, sd_lat=lat)
            t, _b = run(us, plat, plan, trace,
                        dict(psram=64 * KiB, flash=64 * KiB, sd=2 * chunk), chunk=chunk)
            cells.append(fmt_t(t))
        out.append(f"| {chunk // KiB} KiB | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## 5. Speculative verification: g+1 positions per pass "
               "(R+SD, stripe, 64 KiB on all)\n")
    out.append("Weight I/O is unchanged and compute scales by g+1. Expected tokens per pass "
               "with a = 0.6: (1-a^(g+1))/(1-a).\n")
    rates = (3e9, 1e9, 0.3e9)
    out.append("| g | Tokens/pass (a=0.6) | "
               + " | ".join(f"{r / 1e9:g} GMAC/s" for r in rates) + " |")
    out.append("|---|---|" + "---|" * len(rates))
    for g in (0, 1, 2, 4, 8, 16):
        tau = (1 - 0.6 ** (g + 1)) / 0.4
        cells = []
        for rate in rates:
            us, plat, plan, trace, _ = setup("R+SD", "stripe", cfg, rate=rate)
            t, _b = run(us, plat, plan, trace,
                        dict(psram=64 * KiB, flash=64 * KiB, sd=64 * KiB), macs_mul=g + 1)
            cells.append(fmt_t(t / tau))
        out.append(f"| {g} | {tau:.2f} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## 6. Head truncated to 32K vocab (R+SD, stripe, 64 KiB)\n")
    out.append("| Variant | Bytes/token | Time per token | Bound |")
    out.append("|---|---|---|---|")
    for name, vocab in (("full vocab (128256)", 128256),
                        ("truncated vocab (32000)", 32000)):
        c2 = dict(cfg, vocab=vocab)
        us, plat, plan, trace, _ = setup("R+SD", "stripe", c2)
        t, b = run(us, plat, plan, trace, dict(psram=64 * KiB, flash=64 * KiB, sd=64 * KiB))
        nb = sum(r[5] for r in trace)
        out.append(f"| {name} | {nb / MB:.0f} MB | {fmt_t(t)} | {fmt_t(b['parallel'])} |")
    out.append("")
    return "\n".join(out)
