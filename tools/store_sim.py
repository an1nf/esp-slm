#!/usr/bin/env python3
"""Tiered weight store: placement, trace check and event simulator."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import deque
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tbin import FMT_T2, FMT_T3, HEAD_F32, HEAD_I8, LINEARS, read_header
import bytes_sim as bm

MB = 1e6
KiB = 1024
TIERS = ("psram", "flash", "sd")
KV_F32, KV_I8 = 0, 1

BITNET = dict(fmt=FMT_T3, head_fmt=HEAD_I8, dim=2560, n_layers=30, n_heads=20,
              n_kv=5, head_dim=128, inter=6912, vocab=128256, max_seq=4096)


@dataclass(frozen=True)
class Unit:
    id: int
    name: str
    rows: int
    cols: int
    row_bytes: int

    @property
    def bytes(self) -> int:
        return self.rows * self.row_bytes


def row_bytes(fmt: int, n: int) -> int:
    return (n + 3) // 4 if fmt == FMT_T2 else (n + 4) // 5


def head_row_bytes(cfg: dict) -> int:
    return 4 * cfg["dim"] if cfg["head_fmt"] == HEAD_F32 else cfg["dim"] + 4


def units(cfg: dict) -> list[Unit]:
    d, i = cfg["dim"], cfg["inter"]
    q, kv = cfg["n_heads"] * cfg["head_dim"], cfg["n_kv"] * cfg["head_dim"]
    shapes = dict(q=(q, d), k=(kv, d), v=(kv, d), o=(d, q), gate=(i, d),
                  up=(i, d), down=(d, i))
    out = [Unit(0, "head", cfg["vocab"], d, head_row_bytes(cfg))]
    for l in range(cfg["n_layers"]):
        for j, n in enumerate(LINEARS):
            r, c = shapes[n]
            out.append(Unit(1 + 7 * l + j, f"L{l}.{n}", r, c, row_bytes(cfg["fmt"], c)))
    return out


def aux_bytes(cfg: dict) -> int:
    d, q, i = cfg["dim"], cfg["n_heads"] * cfg["head_dim"], cfg["inter"]
    return 4 * (d + cfg["n_layers"] * (2 * d + q + i))


def kv_bytes_per_pos(cfg: dict, kv_fmt: int) -> int:
    kvdim, nkv = cfg["n_kv"] * cfg["head_dim"], cfg["n_kv"]
    per = kvdim + 4 * nkv if kv_fmt == KV_I8 else 4 * kvdim
    return 2 * per * cfg["n_layers"]


def analytic_trace(cfg: dict, pos: int, kv_fmt: int, token: int = 0):
    d, q, i = cfg["dim"], cfg["n_heads"] * cfg["head_dim"], cfg["inter"]
    us = units(cfg)
    hrb = head_row_bytes(cfg)
    per_pos = kv_bytes_per_pos(cfg, kv_fmt) // (2 * cfg["n_layers"])
    t = [(pos, "E", 0, token, token + 1, hrb, 0)]
    for l in range(cfg["n_layers"]):
        t.append((pos, "A", l, 0, 0, 4 * (2 * d + q + i), 0))
        for j in range(7):
            u = us[1 + 7 * l + j]
            t.append((pos, "W", u.id, 0, u.rows, u.bytes, u.rows * u.cols))
            if j == 2:
                t.append((pos, "K", l, 0, pos + 1, (pos + 1) * per_pos * 2,
                          2 * cfg["n_heads"] * (pos + 1) * cfg["head_dim"]))
    t.append((pos, "A", cfg["n_layers"], 0, 0, 4 * d, 0))
    t.append((pos, "H", 0, 0, cfg["vocab"], cfg["vocab"] * hrb, cfg["vocab"] * d))
    return t


def read_trace(path: str):
    with open(path) as f:
        r = csv.DictReader(f)
        return [(int(x["pos"]), x["kind"], int(x["unit"]), int(x["r0"]), int(x["r1"]),
                 int(x["bytes"]), int(x["macs"])) for x in r]


@dataclass
class Channel:
    bw: float
    lat: float = 0.0
    depth: float = 0.0


@dataclass
class Platform:
    name: str
    chans: dict
    caps: dict
    rate: float


def platform(scenario: str, cfg: dict, ctx: int, kv_fmt: int = KV_I8,
             rate: float = bm.COMPUTE_MACS, flash_mb: float = bm.FLASH_FOR_WEIGHTS_MB,
             psram_mb: float = bm.PSRAM_TOTAL_MB, reserve_mb: float = bm.PSRAM_RESERVE_MB,
             sd_lat: float = 0.0) -> Platform:
    sc = next(s for s in bm.SCENARIOS if s.key == scenario)
    psram_cap = (psram_mb - reserve_mb) * MB - ctx * kv_bytes_per_pos(cfg, kv_fmt) \
        - aux_bytes(cfg)
    return Platform(
        scenario,
        dict(psram=Channel(sc.psram * MB), flash=Channel(sc.flash * MB),
             sd=Channel(sc.sd * MB, sd_lat)),
        dict(psram=max(psram_cap, 0.0), flash=flash_mb * MB, sd=math.inf),
        rate)


def targets(total: float, plat: Platform, fixed_psram: float = 0.0,
            serial: bool = False) -> dict:
    bw = {t: plat.chans[t].bw for t in TIERS}
    fix = dict(psram=fixed_psram, flash=0.0, sd=0.0)
    if serial:
        out, need = {}, total
        for t in sorted(TIERS, key=lambda x: -bw[x]):
            take = min(plat.caps[t], need)
            out[t], need = take, need - take
        return out

    def filled(T):
        return sum(min(plat.caps[t], bw[t] * max(0.0, T - fix[t])) for t in TIERS)

    lo, hi = max(fix.values()), max(fix.values()) + 1.0
    while filled(hi) < total:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if filled(mid) >= total else (mid, hi)
    out = {t: min(plat.caps[t], bw[t] * max(0.0, hi - fix[t])) for t in TIERS}
    out["sd"] += total - sum(out.values())
    return out


def place(us: list[Unit], tgt: dict, policy: str) -> dict:
    total = sum(u.bytes for u in us)
    plan = {}
    if policy == "stripe":
        frac = {t: tgt[t] / total for t in TIERS}
        carry = {t: 0.0 for t in TIERS}
        for u in us:
            segs, r = [], 0
            for t in ("psram", "flash"):
                carry[t] += u.rows * frac[t]
                k = min(int(math.floor(carry[t])), u.rows - r)
                carry[t] -= k
                if k > 0:
                    segs.append((t, r, r + k))
                    r += k
            if r < u.rows:
                segs.append(("sd", r, u.rows))
            plan[u.id] = segs
    elif policy == "fill":
        left = {t: tgt[t] for t in ("psram", "flash")}
        for u in sorted(us, key=lambda x: (x.id == 0, x.id)):
            segs, r = [], 0
            for t in ("psram", "flash"):
                k = min(int(left[t] // u.row_bytes), u.rows - r)
                if k > 0:
                    segs.append((t, r, r + k))
                    r += k
                    left[t] -= k * u.row_bytes
            if r < u.rows:
                segs.append(("sd", r, u.rows))
            plan[u.id] = segs
    else:
        raise SystemExit(f"unknown policy: {policy}")
    return plan


def write_placement(path: str, plan: dict, header: str = "") -> None:
    with open(path, "w") as f:
        f.write("# tern placement: unit tier row_start row_end\n")
        if header:
            f.write("".join(f"# {l}\n" for l in header.splitlines()))
        for u in sorted(plan):
            for t, r0, r1 in plan[u]:
                f.write(f"{u} {t} {r0} {r1}\n")


def read_placement(path: str) -> dict:
    plan = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, t, r0, r1 = line.split()
            plan.setdefault(int(u), []).append((t, int(r0), int(r1)))
    return {u: sorted(s, key=lambda x: x[1]) for u, s in plan.items()}


def placed_bytes(us: list[Unit], plan: dict) -> dict:
    out = {t: 0 for t in TIERS}
    for u in us:
        for t, r0, r1 in plan.get(u.id, [("psram", 0, u.rows)]):
            out[t] += (r1 - r0) * u.row_bytes
    return out


def steps_of(trace, us: list[Unit], plan: dict, chunk: int,
             order: str = "interleave"):
    by_id = {u.id: u for u in us}
    out = []
    for _pos, kind, unit, r0, r1, nbytes, macs in trace:
        if kind in ("K", "A"):
            n = max(1, math.ceil(nbytes / chunk))
            for k in range(n):
                b = min(chunk, nbytes - k * chunk)
                out.append(("psram", b, macs * b / nbytes if nbytes else 0))
            continue
        u = by_id[unit]
        segs = plan.get(unit, [("psram", 0, u.rows)])
        per_row_macs = macs / (r1 - r0) if r1 > r0 else 0.0
        rows_per = max(1, chunk // u.row_bytes)
        cur = [[t, max(r0, s0), max(r0, s0), min(r1, s1)] for t, s0, s1 in segs
               if min(r1, s1) > max(r0, s0)]
        if order == "sequential":
            for t, a, _r, b in cur:
                r = a
                while r < b:
                    k = min(rows_per, b - r)
                    out.append((t, k * u.row_bytes, k * per_row_macs))
                    r += k
            continue
        while cur:
            i = min(range(len(cur)),
                    key=lambda j: (cur[j][2] - cur[j][1]) / (cur[j][3] - cur[j][1]))
            t, _a, r, b = cur[i]
            k = min(rows_per, b - r)
            out.append((t, k * u.row_bytes, k * per_row_macs))
            cur[i][2] += k
            if cur[i][2] >= b:
                cur.pop(i)
    return out


def simulate(steps, plat: Platform) -> float:
    free = {t: 0.0 for t in TIERS}
    fifo = {t: deque() for t in TIERS}
    held = {t: 0.0 for t in TIERS}
    prev = 0.0
    for t, b, macs in steps:
        ch = plat.chans[t]
        if ch.depth == 0:
            start = max(free[t], prev)
        else:
            room, q = 0.0, fifo[t]
            while q and held[t] + b > ch.depth:
                end, bb = q.popleft()
                held[t] -= bb
                room = max(room, end)
            start = max(free[t], room)
        rend = start + ch.lat + b / ch.bw
        free[t] = rend
        prev = max(rend, prev) + macs / plat.rate
        if ch.depth:
            fifo[t].append((prev, b))
            held[t] += b
    return prev


def bounds(steps, plat: Platform) -> dict:
    io = {t: 0.0 for t in TIERS}
    macs = 0.0
    for t, b, m in steps:
        io[t] += b / plat.chans[t].bw + plat.chans[t].lat
        macs += m
    comp = macs / plat.rate
    return dict(io=io, comp=comp, parallel=max(max(io.values()), comp),
                serial=sum(io.values()) + comp)


def with_depth(plat: Platform, depth: dict) -> Platform:
    ch = {t: Channel(plat.chans[t].bw, plat.chans[t].lat, depth.get(t, 0.0))
          for t in TIERS}
    return Platform(plat.name, ch, plat.caps, plat.rate)


def cfg_from(args) -> dict:
    if args.tbin:
        return read_header(args.tbin)
    if args.model == "bitnet":
        c = dict(BITNET)
        if getattr(args, "vocab", None):
            c["vocab"] = args.vocab
        return c
    raise SystemExit("pass --tbin or --model bitnet")


def cmd_place(args) -> int:
    cfg = cfg_from(args)
    us = units(cfg)
    total = sum(u.bytes for u in us)
    plat = platform(args.scenario, cfg, args.ctx, KV_I8 if args.kv == "int8" else KV_F32,
                    flash_mb=args.flash_mb)
    if args.psram_mb is not None:
        plat.caps["psram"] = args.psram_mb * MB
    fixed = (args.ctx * kv_bytes_per_pos(cfg, KV_I8 if args.kv == "int8" else KV_F32)
             + aux_bytes(cfg)) / plat.chans["psram"].bw
    tgt = targets(total, plat, fixed, args.serial)
    plan = place(us, tgt, args.policy)
    got = placed_bytes(us, plan)
    hdr = (f"scenario {args.scenario}, policy {args.policy}, ctx {args.ctx}\n"
           + ", ".join(f"{t}: target {tgt[t] / MB:.3f} MB, placed {got[t] / MB:.3f} MB"
                       for t in TIERS))
    write_placement(args.out, plan, hdr)
    print(hdr)
    print(f"wrote {args.out}")
    return 0


def cmd_check(args) -> int:
    cfg = read_header(args.tbin)
    kv = KV_I8 if args.kv == "int8" else KV_F32
    got = read_trace(args.trace)
    ok = True
    poss = sorted({r[0] for r in got})
    for p in poss:
        rows = [r for r in got if r[0] == p]
        tok = rows[0][3]
        exp = analytic_trace(cfg, p, kv, tok)
        if rows != exp:
            ok = False
            bad = next(i for i, (a, b) in enumerate(zip(rows, exp)) if a != b) \
                if len(rows) == len(exp) else min(len(rows), len(exp))
            print(f"trace mismatch at pos {p}, row {bad}: "
                  f"{rows[bad] if bad < len(rows) else None} vs "
                  f"{exp[bad] if bad < len(exp) else None}")
    print(f"tern trace = analytic trace at {len(poss)} positions: "
          f"{'OK' if ok else 'FAIL'}")
    if args.place and args.counters:
        us = units(cfg)
        plan = read_placement(args.place)
        per = {t: 0 for t in TIERS}
        for t, b, _m in steps_of([r for r in got if r[1] in ("E", "W", "H")],
                                 us, plan, 1 << 62):
            per[t] += b
        exp_c = [int(x) for x in args.counters.split(",")]
        same = [per[t] for t in TIERS] == exp_c
        print(f"bytes per tier (tern {exp_c}, analytic {[per[t] for t in TIERS]}): "
              f"{'OK' if same else 'FAIL'}")
        ok &= same
    return 0 if ok else 1


def cmd_selftest(_args) -> int:
    cfg = dict(BITNET, n_layers=2, vocab=4096)
    us = units(cfg)
    plat = platform("R", cfg, 64)
    tr = analytic_trace(cfg, 63, KV_I8)
    total = sum(u.bytes for u in us)
    plan = place(us, targets(total, plat), "stripe")
    st = steps_of(tr, us, plan, 16 * KiB)
    b = bounds(st, plat)
    ok = True

    inf = with_depth(plat, {t: math.inf for t in TIERS})
    fast = Platform("x", inf.chans, inf.caps, 1e30)
    t_inf = simulate(st, fast)
    io_max = max(b["io"].values())
    c1 = abs(t_inf - io_max) / io_max < 1e-3
    print(f"  infinite buffer, zero compute: {t_inf * 1e3:.3f} ms = max I/O "
          f"{io_max * 1e3:.3f} ms: {'ok' if c1 else 'FAIL'}")
    z = with_depth(plat, {})
    t0 = simulate(st, z)
    c2 = abs(t0 - b["serial"]) / b["serial"] < 1e-9
    print(f"  zero buffer: {t0 * 1e3:.3f} ms = serial {b['serial'] * 1e3:.3f} ms: "
          f"{'ok' if c2 else 'FAIL'}")
    t_i = simulate(st, inf)
    c3 = b["parallel"] - 1e-12 <= t_i <= b["parallel"] * 1.02
    print(f"  infinite buffer: {t_i * 1e3:.3f} ms, parallel bound "
          f"{b['parallel'] * 1e3:.3f} ms (within 2 %): {'ok' if c3 else 'FAIL'}")
    prev, mono = math.inf, True
    for d in (0, 16 * KiB, 64 * KiB, 256 * KiB, 4 * MB, math.inf):
        t = simulate(st, with_depth(plat, {x: d for x in TIERS}))
        mono &= t <= prev + 1e-12
        prev = t
    print(f"  monotone in buffer size: {'ok' if mono else 'FAIL'}")
    ok = c1 and c2 and c3 and mono
    print("SIMULATOR:", "OK" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("place")
    p.add_argument("--tbin")
    p.add_argument("--model", choices=("bitnet",))
    p.add_argument("--vocab", type=int)
    p.add_argument("--scenario", default="R")
    p.add_argument("--policy", default="stripe", choices=("stripe", "fill"))
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--kv", default="int8", choices=("int8", "f32"))
    p.add_argument("--flash-mb", type=float, default=bm.FLASH_FOR_WEIGHTS_MB)
    p.add_argument("--psram-mb", type=float, help="override PSRAM capacity for weights")
    p.add_argument("--serial", action="store_true")
    p.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tbin", required=True)
    c.add_argument("--trace", required=True)
    c.add_argument("--kv", default="f32", choices=("int8", "f32"))
    c.add_argument("--place")
    c.add_argument("--counters", help="tern bytes per tier: psram,flash,sd")
    sub.add_parser("selftest")
    r = sub.add_parser("report")
    r.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "place":
        return cmd_place(a)
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "selftest":
        return cmd_selftest(a)
    from store_report import report
    text = report()
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
