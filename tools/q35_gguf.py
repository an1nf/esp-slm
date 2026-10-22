"""GGUF reader/writer and PTQ1_0/Q8_0 packing."""
import struct

import numpy as np

F = np.float32
T_F32, T_F16, T_Q8_0, T_I32, T_PTQ1_0 = 0, 1, 8, 26, 143
TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 26: "I32", 30: "BF16", 142: "PQ2_0", 143: "PTQ1_0"}

V_U8, V_I8, V_U16, V_I16, V_U32, V_I32, V_F32, V_BOOL, V_STR, V_ARR, V_U64, V_I64, V_F64 = range(13)
_SCALAR = {V_U8: "<B", V_I8: "<b", V_U16: "<H", V_I16: "<h", V_U32: "<I", V_I32: "<i",
           V_F32: "<f", V_BOOL: "<?", V_U64: "<Q", V_I64: "<q", V_F64: "<d"}


def row_bytes(ttype, n):
    if ttype == T_F32 or ttype == T_I32:
        return 4 * n
    if ttype == T_F16:
        return 2 * n
    if ttype == T_Q8_0:
        assert n % 32 == 0
        return n // 32 * 34
    if ttype == T_PTQ1_0:
        assert n % 128 == 0
        return n // 128 * 28
    if ttype == 142:
        return n // 128 * 34
    if ttype == 30:
        return 2 * n
    raise ValueError(f"tipo {ttype}")


def _order():
    byte = np.zeros(128, np.int64)
    trit = np.zeros(128, np.int64)
    for n in range(5):
        for m in range(16):
            byte[16 * n + m], trit[16 * n + m] = m, n
        for m in range(8):
            byte[80 + 8 * n + m], trit[80 + 8 * n + m] = 16 + m, n
    for n in range(4):
        for h in range(2):
            byte[120 + 2 * n + h], trit[120 + 2 * n + h] = 24 + h, n
    return byte, trit


_BYTE, _TRIT = _order()
_POW3 = np.array([1, 3, 9, 27, 81], np.int64)
LUT5 = ((((np.arange(256)[:, None] * _POW3[None, :]) % 256) * 3) >> 8) - 1


def ptq1_pack(trits, d):
    t = (np.asarray(trits, np.int64) + 1)
    nblk = t.shape[0]
    out = np.zeros((nblk, 28), np.uint8)
    for b in range(24):
        q = np.zeros(nblk, np.int64)
        for n in range(5):
            e = np.where((_BYTE == b) & (_TRIT == n))[0][0]
            q = q * 3 + t[:, e]
        out[:, b] = (q * 256 + 242) // 243
    for h in range(2):
        q = np.zeros(nblk, np.int64)
        for n in range(4):
            q = q * 3 + t[:, 120 + 2 * n + h]
        q = q * 3
        out[:, 24 + h] = (q * 256 + 242) // 243
    out[:, 26:28] = np.asarray(d, np.float32).astype(np.float16).view(np.uint8).reshape(nblk, 2)
    return out


def ptq1_decode(raw, rows, cols):
    nb = cols // 128
    blk = np.frombuffer(raw, np.uint8).reshape(rows * nb, 28)
    trits = LUT5[blk[:, _BYTE], _TRIT].astype(np.int8)
    d = blk[:, 26:28].copy().view(np.float16).astype(np.float32).reshape(rows, nb)
    return trits.reshape(rows, cols), d


def ptq1_quantize(w):
    rows, cols = w.shape
    x = np.asarray(w, np.float32).reshape(-1, 128)
    amax = np.abs(x).max(1)
    d = amax
    idv = np.where(d != 0, F(1) / np.where(d != 0, d, F(1)), F(0)).astype(F)
    y = x * idv[:, None]
    t = np.clip(np.sign(y) * np.floor(np.abs(y) + 0.5), -1, 1).astype(np.int64)
    return ptq1_pack(t, d).reshape(-1).tobytes()


def q8_quantize_w(w):
    x = np.asarray(w, np.float32).reshape(-1, 32)
    amax = np.abs(x).max(1)
    d = (amax / F(127)).astype(F)
    idv = np.where(d != 0, F(1) / np.where(d != 0, d, F(1)), F(0)).astype(F)
    q = np.round(x * idv[:, None]).astype(np.int8)
    out = np.zeros((x.shape[0], 34), np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1).tobytes()


def q8_decode(raw, rows, cols):
    blk = np.frombuffer(raw, np.uint8).reshape(rows * cols // 32, 34)
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, cols // 32)
    q = blk[:, 2:].copy().view(np.int8).reshape(rows, cols)
    return q, d


def q8_act(x):
    xb = np.asarray(x, F).reshape(-1, 32)
    amax = np.abs(xb).max(1)
    d = (amax / F(127)).astype(np.float16).astype(F)
    idv = np.where(amax != 0, F(127) / np.where(amax != 0, amax, F(1)), F(0)).astype(F)
    q = np.rint(xb * idv[:, None]).astype(np.int8)
    return q.reshape(-1), d


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _value(vtype, v):
    if vtype == V_STR:
        return _str(v)
    if vtype == V_ARR:
        atype, items = v
        out = struct.pack("<IQ", atype, len(items))
        for it in items:
            out += _value(atype, it)
        return out
    return struct.pack(_SCALAR[vtype], v)


class Writer:
    def __init__(self, align=32):
        self.kv = []
        self.tensors = []
        self.align = align

    def add(self, key, vtype, value):
        self.kv.append((key, vtype, value))

    def tensor(self, name, ttype, ne, raw):
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        assert len(raw) == row_bytes(ttype, ne[0]) * rows, name
        self.tensors.append((name, ttype, list(ne), raw))

    def write(self, path):
        head = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv))
        for k, t, v in self.kv:
            head += _str(k) + struct.pack("<I", t) + _value(t, v)
        off = 0
        offs = []
        for name, ttype, ne, raw in self.tensors:
            offs.append(off)
            off += len(raw)
            off = (off + self.align - 1) // self.align * self.align
        for (name, ttype, ne, raw), o in zip(self.tensors, offs):
            head += _str(name) + struct.pack("<I", len(ne))
            head += b"".join(struct.pack("<Q", n) for n in ne)
            head += struct.pack("<IQ", ttype, o)
        pad = (-len(head)) % self.align
        with open(path, "wb") as f:
            f.write(head + b"\0" * pad)
            pos = 0
            for (name, ttype, ne, raw), o in zip(self.tensors, offs):
                f.write(b"\0" * (o - pos))
                f.write(raw)
                pos = o + len(raw)


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        assert f.read(4) == b"GGUF", "no es GGUF"
        ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
        assert ver in (2, 3)
        self.kv = {}
        for _ in range(nkv):
            k = self._rstr()
            (t,) = struct.unpack("<I", f.read(4))
            self.kv[k] = self._rval(t)
        self.tensors = {}
        for _ in range(nt):
            name = self._rstr()
            (nd,) = struct.unpack("<I", f.read(4))
            ne = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            ttype, off = struct.unpack("<IQ", f.read(12))
            self.tensors[name] = (ttype, ne, off)
        align = self.kv.get("general.alignment", 32)
        self.data = (f.tell() + align - 1) // align * align

    def _rstr(self):
        (n,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8", errors="replace")

    def _rval(self, t):
        if t == V_STR:
            return self._rstr()
        if t == V_ARR:
            at, n = struct.unpack("<IQ", self.f.read(12))
            if at == V_STR:
                return [self._rstr() for _ in range(n)]
            fmt = _SCALAR[at]
            size = struct.calcsize(fmt)
            return list(np.frombuffer(self.f.read(size * n), np.dtype(fmt)))
        fmt = _SCALAR[t]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def raw(self, name):
        ttype, ne, off = self.tensors[name]
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        n = row_bytes(ttype, ne[0]) * rows
        self.f.seek(self.data + off)
        return ttype, ne, self.f.read(n)

    def floats(self, name):
        ttype, ne, raw = self.raw(name)
        if ttype == T_F32:
            return np.frombuffer(raw, np.float32).copy()
        if ttype == T_F16:
            return np.frombuffer(raw, np.float16).astype(np.float32)
        raise ValueError(f"{name}: tipo {ttype}")
