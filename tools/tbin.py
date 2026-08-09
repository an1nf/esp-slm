""".tbin format: packing, header and synthetic parameters."""
from __future__ import annotations

import struct

import numpy as np

FMT_T2, FMT_T3 = 1, 2
HEAD_F32, HEAD_I8 = 0, 1
VERSION = 1
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")


def row_bytes(fmt: int, n: int) -> int:
    return (n + 3) // 4 if fmt == FMT_T2 else (n + 4) // 5


def pack_matrix(fmt: int, w: np.ndarray) -> np.ndarray:
    assert w.dtype == np.int8 and w.ndim == 2
    assert np.all((w >= -1) & (w <= 1))
    rows, cols = w.shape
    per = 4 if fmt == FMT_T2 else 5
    rb = row_bytes(fmt, cols)
    d = np.ones((rows, rb * per), dtype=np.int32)
    d[:, :cols] = w.astype(np.int32) + 1
    d = d.reshape(rows, rb, per)
    if fmt == FMT_T2:
        mult = np.array([1, 4, 16, 64], dtype=np.int32)
    else:
        mult = np.array([1, 3, 9, 27, 81], dtype=np.int32)
    return (d * mult).sum(axis=-1).astype(np.uint8)


def quant_rows_i8(e: np.ndarray):
    e = np.asarray(e, np.float32)
    m = np.max(np.abs(e), axis=1)
    sc = np.where(m > 0, m / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(e / sc[:, None]), -127, 127).astype(np.int8)
    return q, sc


def head_bytes(emb: np.ndarray, head_fmt: int) -> bytes:
    if head_fmt == HEAD_F32:
        return np.ascontiguousarray(emb, dtype=np.float32).tobytes()
    q, sc = quant_rows_i8(emb)
    rows = np.empty((q.shape[0], 4 + q.shape[1]), np.uint8)
    rows[:, :4] = sc.view(np.uint8).reshape(-1, 4)
    rows[:, 4:] = q.view(np.uint8)
    return rows.tobytes()


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        h = f.read(64)
    assert h[:4] == b"TERN", path
    v = struct.unpack("<10I2fI", h[4:56])
    keys = ("version", "fmt", "dim", "n_layers", "n_heads", "n_kv", "head_dim",
            "inter", "vocab", "max_seq", "rope_theta", "eps", "head_fmt")
    return dict(zip(keys, v))


class _Writer:
    def __init__(self, f):
        self.f = f
        self.off = 0

    def raw(self, b: bytes):
        self.f.write(b)
        self.off += len(b)

    def pad(self, align: int = 16):
        r = (-self.off) % align
        if r:
            self.raw(b"\0" * r)

    def section(self, b: bytes):
        self.raw(b)
        self.pad()


def write_tbin(path: str, cfg: dict, params: dict, fmt: int,
               head_fmt: int = HEAD_F32) -> None:
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32).tobytes()
    with open(path, "wb") as f:
        w = _Writer(f)
        hdr = b"TERN" + struct.pack(
            "<10I2f", VERSION, fmt, cfg["dim"], cfg["n_layers"], cfg["n_heads"],
            cfg["n_kv"], cfg["head_dim"], cfg["inter"], cfg["vocab"],
            cfg["max_seq"], cfg["rope_theta"], cfg["eps"])
        hdr += struct.pack("<I", head_fmt)
        w.raw(hdr)
        w.raw(b"\0" * (64 - len(hdr)))
        w.section(head_bytes(params["tok_emb"], head_fmt))
        w.section(f32(params["final_norm"]))
        for lay in params["layers"]:
            for k in ("attn_norm", "attn_sub_norm", "ffn_norm", "ffn_sub_norm"):
                w.section(f32(lay[k]))
            for name in LINEARS:
                wt, scale = lay[name]
                w.section(struct.pack("<f", float(scale)))
                w.section(pack_matrix(fmt, wt).tobytes())


def random_params(cfg: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    f32 = np.float32
    dim, inter = cfg["dim"], cfg["inter"]
    qdim = cfg["n_heads"] * cfg["head_dim"]
    kvdim = cfg["n_kv"] * cfg["head_dim"]
    shapes = dict(q=(qdim, dim), k=(kvdim, dim), v=(kvdim, dim), o=(dim, qdim),
                  gate=(inter, dim), up=(inter, dim), down=(dim, inter))

    def tern(shape):
        return rng.choice(np.array([-1, 0, 1], np.int8), size=shape,
                          p=[0.3, 0.4, 0.3]).astype(np.int8)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(f32)

    layers = []
    for _ in range(cfg["n_layers"]):
        lay = dict(attn_norm=norm(dim), attn_sub_norm=norm(qdim),
                   ffn_norm=norm(dim), ffn_sub_norm=norm(inter))
        for n in LINEARS:
            lay[n] = (tern(shapes[n]), float(rng.uniform(0.02, 0.2)))
        layers.append(lay)
    return dict(tok_emb=(rng.standard_normal((cfg["vocab"], dim)) * 0.5).astype(f32),
                final_norm=norm(dim), layers=layers)
