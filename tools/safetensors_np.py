"""Minimal safetensors reader/writer."""
from __future__ import annotations

import json
import struct

import numpy as np


def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return (b >> 16).astype(np.uint16)


def read_safetensors(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        raw = np.asarray(mm[base + s: base + e])
        dt, shape = info["dtype"], tuple(info["shape"])
        if dt == "BF16":
            arr = bf16_to_f32(raw.view(np.uint16))
        elif dt == "F32":
            arr = raw.view(np.float32)
        elif dt == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif dt == "U8":
            arr = raw
        elif dt == "I8":
            arr = raw.view(np.int8)
        else:
            raise ValueError(f"{name}: dtype {dt} no soportado")
        out[name] = (arr.reshape(shape), dt)
    return out


def write_safetensors(path: str, tensors: dict) -> None:
    header, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        if dt == "BF16":
            data = f32_to_bf16(arr).tobytes()
        elif dt == "F32":
            data = np.ascontiguousarray(arr, np.float32).tobytes()
        elif dt == "U8":
            data = np.ascontiguousarray(arr, np.uint8).tobytes()
        else:
            raise ValueError(dt)
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def hf_pack(w: np.ndarray) -> np.ndarray:
    rows = w.shape[0]
    rd = (rows + 3) // 4
    u = (w.astype(np.int16) + 1).astype(np.uint8)
    packed = np.zeros((rd,) + w.shape[1:], np.uint8)
    for i in range(4):
        start, end = i * rd, min(i * rd + rd, rows)
        if start >= rows:
            break
        packed[: end - start] |= (u[start:end] << (2 * i)).astype(np.uint8)
    return packed


def hf_unpack(packed: np.ndarray, rows: int) -> np.ndarray:
    rd = packed.shape[0]
    full = np.empty((rd * 4,) + packed.shape[1:], np.int8)
    for i in range(4):
        full[i * rd:(i + 1) * rd] = ((packed >> (2 * i)) & 3).astype(np.int8) - 1
    return full[:rows]
