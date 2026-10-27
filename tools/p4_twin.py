#!/usr/bin/env python3
"""ESP32-P4 twin: memory fit, per-pass time and lanes per pass for Ternary Bonsai 2."""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import store_sim as ss

MB = 1e6
V = 248320
PTQ_BPB = 28 / 128

CFG = {
    "2B": dict(E=2048, Fd=6144, L=24, nh=8, nkv=2, hd=256, Hk=16, Hv=16, d=128, tied=True),
    "4B": dict(E=2560, Fd=9216, L=32, nh=16, nkv=4, hd=256, Hk=16, Hv=32, d=128, tied=True),
    "9B": dict(E=4096, Fd=12288, L=32, nh=16, nkv=4, hd=256, Hk=16, Hv=32, d=128, tied=False),
}
K_EMBD, K_HEAD, K_ATTN, K_GDN, K_GDN_F32, K_FFN = range(6)


@dataclass
class P4:
    psram_bw: float = 160.0
    flash_bw: float = 30.0
    sd_bw: float = 40.0
    sd_lat: float = 0.3e-3
    psram_mb: float = 32.0
    reserve_mb: float = 3.0
    flash_mb: float = 14.0
    decode_mbps: float = 200.0
    lane_gmacs: float = 6.4
    fp_gflops: float = 0.4
    chunk: int = 64 * 1024
    hd_block: int = 512
    depth: dict = None

    def __post_init__(self):
        if self.depth is None:
            self.depth = dict(psram=64 * 1024, flash=64 * 1024, sd=256 * 1024)


@dataclass
class Variant:
    model: str = "2B"
    T: int = 1
    ctx: int = 1024
    state_bytes: float = 2.0
    kv_bytes: float = 2.0
    head_rows: int = V
    core: str = "seq"
    wy_rank: int = 8
    wy_flush: int = 4


@dataclass
class Mat:
    kind: int
    layer: int
    rows: int
    cols: int
    row_bytes: float

    @property
    def bytes(self):
        return self.rows * self.row_bytes


def mats(c: dict, head_rows: int = V):
    E, Fd, hd = c["E"], c["Fd"], c["hd"]
    kd, vd = c["Hk"] * c["d"], c["Hv"] * c["d"]
    cd = 2 * kd + vd
    out = []
    t = lambda k, l, r, cols: Mat(k, l, r, cols, cols * PTQ_BPB)
    for l in range(c["L"]):
        if (l + 1) % 4 != 0:
            out += [("rot", E), t(K_GDN, l, cd, E), t(K_GDN, l, vd, E),
                    Mat(K_GDN_F32, l, c["Hv"], E, 4 * E), Mat(K_GDN_F32, l, c["Hv"], E, 4 * E),
                    ("gdn", l), ("rot", vd), t(K_GDN, l, E, vd)]
        else:
            out += [("rot", E), t(K_ATTN, l, 2 * c["nh"] * hd, E), t(K_ATTN, l, c["nkv"] * hd, E),
                    t(K_ATTN, l, c["nkv"] * hd, E), ("attn", l), ("rot", c["nh"] * hd),
                    t(K_ATTN, l, E, c["nh"] * hd)]
        out += [("rot", E), t(K_FFN, l, Fd, E), t(K_FFN, l, Fd, E), ("rot", Fd), t(K_FFN, l, E, Fd)]
    out += [("rot", E), t(K_HEAD, -1, head_rows, E)]
    return out


def memory(c: dict, v: Variant):
    nrec = c["L"] * 3 // 4
    natt = c["L"] - nrec
    kd, vd = c["Hk"] * c["d"], c["Hv"] * c["d"]
    state = nrec * c["Hv"] * c["d"] * c["d"] * v.state_bytes
    conv = nrec * 3 * (2 * kd + vd) * 4
    kv = v.ctx * natt * 2 * c["nkv"] * c["hd"] * v.kv_bytes
    aux = c["L"] * 2 * c["E"] * 4 + nrec * (2 * c["Hv"] + c["d"]) * 4 + nrec * (2 * kd + vd) * 16
    out = dict(state=state, conv=conv, kv=kv, aux=aux)
    if v.core == "ro8":
        out["delta"] = nrec * v.wy_rank * ((c["Hv"] + c["Hk"]) * c["d"] * 2 + c["Hv"] * 8)
    return out


def core_cost(c: dict, kind: str, v: Variant, p: P4, width: int = 0):
    if kind == "rot":
        if not p.hd_block:
            return 0.0, 0.0
        n = width
        return 0.0, v.T * n * (math.log2(p.hd_block) + 2) / (p.fp_gflops * 1e9)
    if kind == "gdn" and v.core == "ro8":
        d, H, N, r = c["d"], c["Hv"], v.T, v.wy_rank
        macs = H * 2 * N * d * d
        flops = H * (N * N * 2 * d + 2 * r * N * 2 * d + (r + 1) * d * d / v.wy_flush) \
            + N * (2 * c["Hk"] + H) * d * 12
        byt = H * d * d * 1.0 * (1 + 1.0 / v.wy_flush)
        return byt, macs / (p.lane_gmacs * 1e9) + flops / (p.fp_gflops * 1e9)
    if kind == "gdn":
        st = c["Hv"] * c["d"] * c["d"]
        flops = v.T * (st * 8 + (2 * c["Hk"] + c["Hv"]) * c["d"] * 12)
        return 2 * st * v.state_bytes, flops / (p.fp_gflops * 1e9)
    nkv_b = 2 * c["nkv"] * c["hd"] * v.kv_bytes
    kvread = v.ctx * nkv_b + v.T * nkv_b
    if v.core == "ro8":
        macs = v.T * (v.ctx + v.T) * c["nh"] * c["hd"] * 2
        flops = v.T * (v.ctx + v.T) * c["nh"] * 6 + v.T * c["nh"] * c["hd"] * 20
        return kvread, macs / (p.lane_gmacs * 1e9) + flops / (p.fp_gflops * 1e9)
    flops = v.T * (v.ctx + v.T) * c["nh"] * c["hd"] * 4 + v.T * c["nh"] * c["hd"] * 20
    return kvread, flops / (p.fp_gflops * 1e9)


def plan_tiers(c: dict, v: Variant, p: P4):
    seq = mats(c, v.head_rows)
    ms = [m for m in seq if isinstance(m, Mat)]
    W = sum(m.bytes for m in ms)
    mem = memory(c, v)
    fixed = sum(mem.values())
    psram_cap = p.psram_mb * MB - p.reserve_mb * MB - fixed
    busy = sum(core_cost(c, x[0], v, p, x[1])[0] for x in seq if isinstance(x, tuple)) / (p.psram_bw * MB)
    plat = ss.Platform("p4", dict(psram=ss.Channel(p.psram_bw * MB),
                                  flash=ss.Channel(p.flash_bw * MB),
                                  sd=ss.Channel(p.sd_bw * MB, p.sd_lat)),
                       dict(psram=max(psram_cap, 0.0), flash=p.flash_mb * MB, sd=math.inf),
                       1.0)
    tgt = ss.targets(W, plat, fixed_psram=busy)
    return seq, W, mem, psram_cap, tgt, plat


def steps(c: dict, v: Variant, p: P4):
    seq, W, mem, psram_cap, tgt, plat = plan_tiers(c, v, p)
    frac = {t: tgt[t] / W for t in ss.TIERS}
    carry = {t: 0.0 for t in ss.TIERS}
    out = []
    for _ in range(v.T):
        out.append(("sd", c["E"] * PTQ_BPB, 0.0))
    for m in seq:
        if isinstance(m, tuple):
            b, sec = core_cost(c, m[0], v, p, m[1])
            n = max(1, math.ceil(b / p.chunk))
            for k in range(n):
                out.append(("psram", min(p.chunk, b - k * p.chunk), sec / n))
            continue
        segs, r = [], 0
        for t in ("psram", "flash"):
            carry[t] += m.rows * frac[t]
            k = min(int(math.floor(carry[t])), m.rows - r)
            carry[t] -= k
            if k > 0:
                segs.append([t, r, r, r + k])
                r += k
        if r < m.rows:
            segs.append(["sd", r, r, m.rows])
        rows_per = max(1, int(p.chunk // m.row_bytes))
        while segs:
            i = min(range(len(segs)), key=lambda j: (segs[j][2] - segs[j][1]) / (segs[j][3] - segs[j][1]))
            t, a, cur, b = segs[i]
            k = min(rows_per, b - cur)
            byt = k * m.row_bytes
            if m.kind == K_GDN_F32:
                sec = k * m.cols * v.T * 2 / (p.fp_gflops * 1e9)
            else:
                sec = byt / (p.decode_mbps * MB) + k * m.cols * v.T / (p.lane_gmacs * 1e9)
            out.append((t, byt, sec))
            segs[i][2] += k
            if segs[i][2] >= b:
                segs.pop(i)
    return out, dict(W=W, mem=mem, psram_cap=psram_cap, tgt=tgt, plat=plat)


def run_pass(v: Variant, p: P4):
    c = CFG[v.model]
    st, info = steps(c, v, p)
    plat = ss.with_depth(info["plat"], p.depth)
    t = ss.simulate(st, plat)
    io = {k: 0.0 for k in ss.TIERS}
    comp = 0.0
    for tier, b, sec in st:
        io[tier] += b / plat.chans[tier].bw + plat.chans[tier].lat
        comp += sec
    info.update(time=t, io=io, comp=comp, fits=info["psram_cap"] >= 0)
    return info


def tokens_per_pass(a: float, gamma: int) -> float:
    if gamma <= 0:
        return 1.0
    return (1 - a ** (gamma + 1)) / (1 - a) if a < 1 else gamma + 1.0


def fmt_mem(c, v):
    m = memory(c, v)
    return " + ".join(f"{k} {x / MB:.1f}" for k, x in m.items())


def report(out_path: str | None):
    L = []
    P = lambda s="": L.append(s)
    p0 = P4()
    P("# ESP32-P4 twin: Ternary Bonsai 2 on q35")
    P()
    P("Per-pass times from the event simulator (local R1 model) over the exact q35 matrix "
      "sequence, `stripe` placement and bandwidth-optimal tier split. Default P4 parameters "
      "([M] measured, [H] assumed):")
    P()
    P(f"- PSRAM {p0.psram_bw:.0f} MB/s [M], flash {p0.flash_bw:.0f} MB/s [H], SD variable "
      f"(16 [M] at 40 MHz; 40 and 80 [H] in UHS-I), SD latency {p0.sd_lat * 1e3:.1f} ms per read [H].")
    P(f"- PTQ1_0 decode {p0.decode_mbps:.0f} MB/s [H], int8 dot {p0.lane_gmacs} GMAC/s per "
      f"lane [H], scalar float {p0.fp_gflops} GFLOP/s [H].")
    P(f"- PSRAM {p0.psram_mb:.0f} MB with {p0.reserve_mb:.0f} MB reserved; flash for weights "
      f"{p0.flash_mb:.0f} MB; prefetch buffers PSRAM/flash 64 KiB, SD 256 KiB.")
    P()

    P("## 1. Memory: what fits in PSRAM")
    P()
    P("Free PSRAM for weights = 32 - 3 - state - conv - KV - aux (MB); aux = norms, GDN params "
      "and conv weights. Negative = does not fit.")
    P()
    P("| Model | State | KV | ctx 512 | ctx 1024 | ctx 2048 |")
    P("|---|---|---|---|---|---|")
    for mo in ("2B", "4B", "9B"):
        c = CFG[mo]
        for sb, sn in ((4, "f32"), (2, "f16"), (1, "int8")):
            for kb, kn in ((2, "f16"), (1, "int8")):
                cells = []
                for ctx in (512, 1024, 2048):
                    v = Variant(mo, 1, ctx, sb, kb)
                    m = memory(c, v)
                    free = (p0.psram_mb - p0.reserve_mb) * MB - sum(m.values())
                    cells.append(f"{free / MB:.1f}" if free >= 0 else f"**{free / MB:.1f}**")
                P(f"| {mo} | {sn} | {kn} | " + " | ".join(cells) + " |")
    P()

    P("## 2. One pass (T = 1) vs SD speed")
    P()
    P("State and KV f16 for 2B, int8 for 4B/9B (what fits at ctx 1024).")
    P()
    P("| Model | Head vocab | SD (MB/s) | s/pass | SD I/O | PSRAM I/O | Compute | max(I/O, compute) |")
    P("|---|---|---|---|---|---|---|---|")
    base = {}
    for mo in ("2B", "4B", "9B"):
        sb, kb = (2, 2) if mo == "2B" else (1, 1)
        for hr in (V, 32768):
            for sd in (16, 40, 80):
                p = replace(P4(), sd_bw=sd)
                v = Variant(mo, 1, 1024, sb, kb, hr)
                r = run_pass(v, p)
                base[(mo, hr, sd)] = r["time"]
                bound = max(max(r["io"].values()), r["comp"])
                P(f"| {mo} | {hr // 1000}K | {sd} | **{r['time']:.1f}** | {r['io']['sd']:.1f} | "
                  f"{r['io']['psram']:.2f} | {r['comp']:.2f} | {bound:.1f} |")
    P()

    P("## 3. Free lanes: verifying T tokens in one pass")
    P()
    P("s/pass with T tokens (SD 40 MB/s, full head). While I/O dominates, each extra token "
      "only adds compute hidden behind the read.")
    P()
    Ts = (1, 2, 4, 8, 16, 32)
    P("| Model | " + " | ".join(f"T={t}" for t in Ts) + " |")
    P("|---|" + "---|" * len(Ts))
    lanes = {}
    for mo in ("2B", "4B", "9B"):
        sb, kb = (2, 2) if mo == "2B" else (1, 1)
        row = []
        for T in Ts:
            r = run_pass(Variant(mo, T, 1024, sb, kb), replace(P4(), sd_bw=40))
            lanes[(mo, T)] = r["time"]
            row.append(f"{r['time']:.1f}")
        P(f"| {mo} | " + " | ".join(row) + " |")
    P()

    P("## 4. Effective s/token with drafts")
    P()
    P("Chain of g = T-1 drafts, i.i.d. acceptance a; tokens per pass "
      "tau = (1 - a^(g+1))/(1 - a). Drafts are free (n-gram or context lookup). SD 40 MB/s.")
    P()
    P("| Model | a | T=1 | T=2 | T=4 | T=8 | T=16 |")
    P("|---|---|---|---|---|---|---|")
    for mo in ("2B", "4B", "9B"):
        for a in (0.5, 0.7, 0.85):
            cells = []
            for T in (1, 2, 4, 8, 16):
                cells.append(f"{lanes[(mo, T)] / tokens_per_pass(a, T - 1):.1f}")
            P(f"| {mo} | {a} | " + " | ".join(cells) + " |")
    P()

    P("## 5. What limits multi-token verification (2B, T = 8, SD 40)")
    P()
    P("At 40 MB/s, weight decode (100-400 MB/s [H]) and the per-lane dot (3.2-12.8 GMAC/s [H]) "
      "hide behind the read. What costs is non-matrix float work (GDN recurrence and attention), "
      "which stalls the pipeline while the SD buffer is full.")
    P()
    P("| Decode (MB/s) | GMAC/s per lane | s/pass |")
    P("|---|---|---|")
    for dec, lg in ((100, 3.2), (200, 6.4), (400, 12.8)):
        r = run_pass(Variant("2B", 8, 1024, 2, 2), replace(P4(), sd_bw=40, decode_mbps=dec, lane_gmacs=lg))
        P(f"| {dec} | {lg} | {r['time']:.1f} |")
    P()
    P("| Float GFLOP/s | SD buffer | s/pass T=1 | s/pass T=8 | s/pass T=16 |")
    P("|---|---|---|---|---|")
    for fp in (0.2, 0.4, 0.8):
        for dep in (256 * 1024, 1 << 20, 4 << 20):
            cells = []
            for T in (1, 8, 16):
                p = replace(P4(), sd_bw=40, fp_gflops=fp)
                p.depth = dict(psram=64 * 1024, flash=64 * 1024, sd=dep)
                cells.append(f"{run_pass(Variant('2B', T, 1024, 2, 2), p)['time']:.1f}")
            P(f"| {fp} | {dep // 1024} KiB | " + " | ".join(cells) + " |")
    P()
    txt = "\n".join(L)
    if out_path:
        with open(out_path, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return base, lanes


def combos(out_path=None):
    L = []
    P = lambda x="": L.append(x)
    meas = {"2B": [("no drafts", 1.0, 1), ("mix 8, mean", 1.44, 5), ("mix 16, mean", 1.61, 9),
                   ("mix 16, summary", 2.28, 12)],
            "4B": [("no drafts", 1.0, 1), ("mix 8, mean", 1.49, 5), ("mix 8, summary", 2.21, 7)],
            "9B": [("no drafts", 1.0, 1), ("mix 8, 4B mean", 1.49, 5),
                   ("mix 8, 4B summary", 2.21, 7)]}
    chans = [("SD 16 [M]", 16), ("SD 40 [H]", 40), ("SD 40 + USB 20 [H]", 60), ("SD 80 [H]", 80),
             ("SD 80 + USB 20 [H]", 100)]
    P("# Configurations for the 5-15 s/token target")
    P()
    P("s/token = s/pass(T) / tau, with tau and T measured on the PC. State and KV f16 for 2B, int8 "
      "for 4B. ctx 1024 for 2B, 512 for 4B. Full head (248K) or truncated to the 32K most frequent "
      "tokens. **Bold**: within target.")
    P()
    for mo in ("2B", "4B", "9B"):
        sb = 2 if mo == "2B" else 1
        ctx = 1024 if mo == "2B" else 512
        P(f"## {mo}" + (" (tau assumed equal to 4B)" if mo == "9B" else ""))
        P()
        P("| Slow channel | Head | " + " | ".join(f"{n} (tau {t:.2f})" for n, t, _ in meas[mo]) + " |")
        P("|---|---|" + "---|" * len(meas[mo]))
        for cn, bw in chans:
            for hr, hn in ((V, "248K"), (32768, "32K")):
                cells = []
                for n, tau, T in meas[mo]:
                    tp = run_pass(Variant(mo, T, ctx, sb, sb, hr), replace(P4(), sd_bw=bw))["time"]
                    x = tp / tau
                    cells.append(f"**{x:.1f}**" if 5 <= x <= 15 else (f"*{x:.1f}*" if x < 5 else f"{x:.1f}"))
                P(f"| {cn} | {hn} | " + " | ".join(cells) + " |")
        P()
    txt = "\n".join(L)
    if out_path:
        with open(out_path, "w") as f:
            f.write(txt + "\n")
    print(txt)


def check(model: str, trace: str) -> int:
    c = CFG[model]
    want = [m for m in mats(c) if isinstance(m, Mat)]
    got = []
    with open(trace) as f:
        for line in f:
            p = line.split()
            if p and p[0] == "W":
                kind, layer, rows, cols, byt = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])
                got.append((kind, layer, rows, cols, byt))
    got = got[:len(want)]
    bad = 0
    for i, (m, g) in enumerate(zip(want, got)):
        if (m.kind, m.layer, m.rows, m.cols, round(m.bytes)) != g:
            bad += 1
            if bad <= 5:
                print(f"  {i}: twin {(m.kind, m.layer, m.rows, m.cols, round(m.bytes))} q35 {g}")
    tot_w = sum(round(m.bytes) for m in want)
    tot_g = sum(g[4] for g in got)
    print(f"{len(want)} matrices; mismatched: {bad}; bytes twin {tot_w} q35 {tot_g}")
    return 1 if bad or len(got) != len(want) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--out")
    pp = sub.add_parser("pass")
    pp.add_argument("--model", default="2B", choices=sorted(CFG))
    pp.add_argument("--T", type=int, default=1)
    pp.add_argument("--ctx", type=int, default=1024)
    pp.add_argument("--state", type=float, default=2.0)
    pp.add_argument("--kv", type=float, default=2.0)
    pp.add_argument("--head", type=int, default=V)
    pp.add_argument("--sd", type=float, default=40.0)
    pp.add_argument("--decode", type=float, default=200.0)
    pp.add_argument("--lane", type=float, default=6.4)
    cb = sub.add_parser("combos")
    cb.add_argument("--out")
    ck = sub.add_parser("check")
    ck.add_argument("--model", default="2B", choices=sorted(CFG))
    ck.add_argument("--trace", required=True)
    a = ap.parse_args()
    if a.cmd == "report":
        report(a.out)
    elif a.cmd == "combos":
        combos(a.out)
    elif a.cmd == "pass":
        r = run_pass(Variant(a.model, a.T, a.ctx, a.state, a.kv, a.head),
                     replace(P4(), sd_bw=a.sd, decode_mbps=a.decode, lane_gmacs=a.lane))
        print(f"s/pass {r['time']:.2f}; I/O { {k: round(x, 2) for k, x in r['io'].items()} }; "
              f"compute {r['comp']:.2f}; free PSRAM for weights {r['psram_cap'] / MB:.1f} MB; "
              f"split { {k: round(x / MB, 1) for k, x in r['tgt'].items()} } MB")
    else:
        return check(a.model, a.trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
