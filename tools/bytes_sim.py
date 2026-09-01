#!/usr/bin/env python3
"""Analytic bytes-per-token model for ternary LLMs on the ESP32-P4."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

MB = 1e6
INF = float("inf")


@dataclass(frozen=True)
class Model:
    name: str
    hidden: int
    inter: int
    ffn_mats: int
    vocab: int
    tied: bool
    full_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    attn_out_gate: bool
    lin_layers: int = 0
    lin_kh: int = 0
    lin_vh: int = 0
    lin_dk: int = 0
    lin_dv: int = 0
    conv_k: int = 4

    def attn_params(self) -> int:
        q_out = self.n_heads * self.head_dim * (2 if self.attn_out_gate else 1)
        kv_out = 2 * self.n_kv * self.head_dim
        o = self.n_heads * self.head_dim * self.hidden
        return self.hidden * (q_out + kv_out) + o

    def gdn_params(self) -> int:
        if not self.lin_layers:
            return 0
        qk = 2 * self.lin_kh * self.lin_dk
        vz = 2 * self.lin_vh * self.lin_dv
        in_proj = self.hidden * (qk + vz)
        ba = self.hidden * 2 * self.lin_vh
        conv = (qk + self.lin_vh * self.lin_dv) * self.conv_k
        out = self.lin_vh * self.lin_dv * self.hidden
        return in_proj + ba + conv + out

    def ffn_params(self) -> int:
        return self.ffn_mats * self.hidden * self.inter

    def body_params(self) -> int:
        return (self.full_layers * (self.attn_params() + self.ffn_params())
                + self.lin_layers * (self.gdn_params() + self.ffn_params()))

    def emb_params(self) -> int:
        return self.vocab * self.hidden

    def kv_bytes_per_pos(self, kv_bytes: float) -> float:
        return self.full_layers * 2 * self.n_kv * self.head_dim * kv_bytes

    def rec_state_values(self) -> int:
        return self.lin_layers * self.lin_vh * self.lin_dk * self.lin_dv


BITNET_2B4T = Model(
    name="BitNet b1.58 2B4T", hidden=2560, inter=6912, ffn_mats=3,
    vocab=128256, tied=True, full_layers=30, n_heads=20, n_kv=5,
    head_dim=128, attn_out_gate=False)

QWEN35_2B = Model(
    name="Qwen3.5-2B (Bonsai 2 2B base)", hidden=2048, inter=6144,
    ffn_mats=3, vocab=248320, tied=True, full_layers=6, n_heads=8, n_kv=2,
    head_dim=256, attn_out_gate=True, lin_layers=18, lin_kh=16, lin_vh=16,
    lin_dk=128, lin_dv=128)

QWEN35_08B = Model(
    name="Qwen3.5-0.8B (hypothetical ternary)", hidden=1024, inter=3584,
    ffn_mats=3, vocab=248320, tied=True, full_layers=6, n_heads=8, n_kv=2,
    head_dim=256, attn_out_gate=True, lin_layers=18, lin_kh=16, lin_vh=16,
    lin_dk=128, lin_dv=128)


@dataclass(frozen=True)
class Variant:
    label: str
    model: Model
    bpw_body: float
    bpw_head: float
    head_vocab: int | None = None
    state_bytes: float = 4.0
    kv_bytes: float = 1.0

    def head_rows(self) -> int:
        return self.head_vocab or self.model.vocab


VARIANTS = [
    Variant("BitNet 2B, bf16 head", BITNET_2B4T, 1.6875, 16.0),
    Variant("BitNet 2B, int8 head", BITNET_2B4T, 1.6875, 8.0),
    Variant("BitNet 2B, 4-bit head, 32K vocab", BITNET_2B4T, 1.6875, 4.5,
            head_vocab=32_000),
    Variant("Bonsai 2 2B, fp32 state", QWEN35_2B, 1.75, 1.75,
            state_bytes=4.0),
    Variant("Bonsai 2 2B, int16 state", QWEN35_2B, 1.75, 1.75,
            state_bytes=2.0),
    Variant("Bonsai 2 2B, int16 state, 32K vocab", QWEN35_2B, 1.75, 1.75,
            head_vocab=32_000, state_bytes=2.0),
    Variant("0.8B ternary, int16 state", QWEN35_08B, 1.75, 1.75,
            state_bytes=2.0),
    Variant("0.8B ternary, int16 state, 32K vocab", QWEN35_08B, 1.75, 1.75,
            head_vocab=32_000, state_bytes=2.0),
]


@dataclass(frozen=True)
class Scenario:
    key: str
    desc: str
    psram: float
    flash: float
    sd: float
    usb: float


SCENARIOS = [
    Scenario("M", "conservative (worst measured)", 110, 8, 3.5, 7.5),
    Scenario("R", "realistic (best measured)", 160, 30, 16.2, 7.5),
    Scenario("R+SD", "realistic with fast SD (assumed)", 160, 30, 60, 7.5),
    Scenario("T", "interface ceiling", 800, 60, 104, 60),
]

PSRAM_TOTAL_MB = 32.0
PSRAM_RESERVE_MB = 2.0
FLASH_FOR_WEIGHTS_MB = 28.0
COMPUTE_MACS = 3.0e9
POWER_W = 0.55


@dataclass
class PassCost:
    w_bytes: float
    body_bytes: float
    head_bytes: float
    state_traffic: float
    psram_free: float
    macs: float
    fits: bool
    max_ctx: int


def pass_cost(v: Variant, ctx: int) -> PassCost:
    m = v.model
    body = m.body_params() * v.bpw_body / 8
    head = v.head_rows() * m.hidden * v.bpw_head / 8
    rec_state = m.rec_state_values() * v.state_bytes
    kv = ctx * m.kv_bytes_per_pos(v.kv_bytes)
    traffic = 2 * rec_state + kv
    usable = (PSRAM_TOTAL_MB - PSRAM_RESERVE_MB) * MB
    psram_free = usable - rec_state - kv
    macs = m.body_params() + v.head_rows() * m.hidden
    kv_pos = m.kv_bytes_per_pos(v.kv_bytes)
    max_ctx = int((usable - rec_state) // kv_pos) if kv_pos else 10**9
    return PassCost(body + head, body, head, traffic, max(psram_free, 0.0),
                    macs, psram_free >= 0, max(max_ctx, 0))


def io_time(pc: PassCost, sc: Scenario, serial: bool) -> tuple[float, dict]:
    tiers = [
        ("PSRAM", pc.psram_free, sc.psram * MB, pc.state_traffic / (sc.psram * MB)),
        ("flash", FLASH_FOR_WEIGHTS_MB * MB, sc.flash * MB, 0.0),
        ("SD", INF, sc.sd * MB, 0.0),
        ("USB", INF, sc.usb * MB, 0.0),
    ]
    need = pc.w_bytes
    if serial:
        t = tiers[0][3]
        alloc = {}
        for name, cap, bw, _ in sorted(tiers, key=lambda x: -x[2]):
            take = min(cap, need)
            alloc[name] = take
            t += take / bw
            need -= take
            if need <= 0:
                break
        return t, alloc

    def filled(T: float) -> float:
        return sum(min(cap, bw * max(0.0, T - fix)) for _, cap, bw, fix in tiers)

    lo = max(f for *_, f in tiers)
    hi = lo + 1.0
    while filled(hi) < need:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if filled(mid) >= need:
            hi = mid
        else:
            lo = mid
    alloc = {n: min(c, b * max(0.0, hi - f)) for n, c, b, f in tiers}
    return hi, alloc


GAMMA_MAX = 16
ALPHAS = (0.3, 0.6, 0.8)


def best_speculative(t_io: float, macs: float, alpha: float, overlap: bool,
                     gamma_max: int = GAMMA_MAX) -> tuple[float, int, float]:
    best = (0.0, 0, 1.0)
    for g in range(0, gamma_max + 1):
        p = g + 1
        tau = p if alpha >= 1 else (1 - alpha ** p) / (1 - alpha) if alpha else 1.0
        t_c = p * macs / COMPUTE_MACS
        t = max(t_io, t_c) if overlap else t_io + t_c
        tps = tau / t
        if tps > best[0]:
            best = (tps, g, tau)
    return best


def fmt_mb(x: float) -> str:
    return f"{x / MB:,.1f}".replace(",", " ")


def fmt_t(s: float) -> str:
    if s >= 1:
        return f"{s:.1f} s"
    return f"{s * 1000:.0f} ms"


def report(ctx: int, serial: bool, overlap: bool) -> str:
    out = []
    out.append(f"# Bytes per token on the ESP32-P4 (ctx {ctx} tokens)\n")
    out.append("Assumptions: 32 MB PSRAM (2 MB reserved), "
               f"{FLASH_FOR_WEIGHTS_MB:.0f} MB flash for weights, "
               f"compute {COMPUTE_MACS / 1e9:.0f} GMAC/s, power {POWER_W} W, "
               f"channels {'serial' if serial else 'parallel'}, "
               f"I/O and compute {'overlapped' if overlap else 'not overlapped'}.\n")

    out.append("## 1. Model anatomy\n")
    out.append("| Model | Body | Embeddings/head | % head | "
               "KV per position (int8) | Recurrent state (values) |")
    out.append("|---|---|---|---|---|---|")
    for m in (BITNET_2B4T, QWEN35_2B, QWEN35_08B):
        body, emb = m.body_params(), m.emb_params()
        out.append(f"| {m.name} | {body / 1e6:,.0f} M | {emb / 1e6:,.0f} M | "
                   f"{100 * emb / (body + emb):.0f} % | "
                   f"{m.kv_bytes_per_pos(1) / 1024:.1f} KB | "
                   f"{m.rec_state_values() / 1e6:.2f} M |")
    out.append("")

    out.append("## 2. Bytes per pass\n")
    out.append("| Variant | Body | Head | State+KV traffic (PSRAM) | "
               "Free PSRAM for weights | Total weights | Max ctx in PSRAM |")
    out.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        pc = pass_cost(v, ctx)
        free = f"{fmt_mb(pc.psram_free)} MB" if pc.fits else "**does not fit**"
        out.append(f"| {v.label} | {fmt_mb(pc.body_bytes)} MB | "
                   f"{fmt_mb(pc.head_bytes)} MB | {fmt_mb(pc.state_traffic)} MB | "
                   f"{free} | {fmt_mb(pc.w_bytes)} MB | "
                   + f"{pc.max_ctx:,}".replace(",", " ") + " tokens |")
    out.append("")

    out.append("## 3. Time per token\n")
    out.append("Speculative decoding with weight-free drafts (context n-grams or "
               "suffixes, cost ~0); a = per-token acceptance [H]. "
               "g* = optimal draft length.\n")
    hdr_a = " | ".join(f"a={a:.1f}" for a in ALPHAS)
    out.append("| Variant | Scen. | I/O per pass | Compute per position | "
               f"No speculation | {hdr_a} | J/token (a=0.6) |")
    out.append("|---|---|---|---|---|" + "---|" * len(ALPHAS) + "---|")
    for v in VARIANTS:
        pc = pass_cost(v, ctx)
        if not pc.fits:
            out.append(f"| {v.label} | - | state + KV do not fit in PSRAM "
                       f"at {ctx} tokens (max {pc.max_ctx}) |" + " |" * 6)
            continue
        for sc in SCENARIOS:
            t_io, _ = io_time(pc, sc, serial)
            t_c = pc.macs / COMPUTE_MACS
            t0 = max(t_io, t_c) if overlap else t_io + t_c
            cells = []
            for a in ALPHAS:
                tps, g, _tau = best_speculative(t_io, pc.macs, a, overlap)
                cells.append(f"{fmt_t(1 / tps)} (g*={g})")
            tps6, *_ = best_speculative(t_io, pc.macs, 0.6, overlap)
            mj = POWER_W / tps6
            out.append(f"| {v.label} | {sc.key} | {fmt_t(t_io)} | {fmt_t(t_c)} | "
                       f"{fmt_t(t0)} | " + " | ".join(cells) + f" | {mj:.2f} |")
    out.append("")

    out.append("## 4. Weight split per tier (scenario R)\n")
    out.append("| Variant | PSRAM | flash | SD | USB |")
    out.append("|---|---|---|---|---|")
    sc_r = next(s for s in SCENARIOS if s.key == "R")
    for v in VARIANTS:
        pc = pass_cost(v, ctx)
        _, alloc = io_time(pc, sc_r, serial)
        out.append(f"| {v.label} | " + " | ".join(
            f"{fmt_mb(alloc.get(n, 0.0))} MB" for n in ("PSRAM", "flash", "SD", "USB"))
            + " |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", type=int, default=512, help="context length")
    ap.add_argument("--serial", action="store_true",
                    help="serial instead of parallel channels")
    ap.add_argument("--no-overlap", action="store_true",
                    help="no I/O-compute overlap")
    args = ap.parse_args()
    print(report(args.ctx, args.serial, not args.no_overlap))


if __name__ == "__main__":
    main()
