"""Q35R logit files: load, compare and KL."""
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"Q35R", path
        n, V, kv = struct.unpack("<III", f.read(12))
        ids = np.frombuffer(f.read(4 * n), np.int32)
        lg = np.frombuffer(f.read(4 * n * V), np.float32).reshape(n, V)
    return ids, lg


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def compare(a, b, verbose=True):
    ia, la = load(a)
    ib, lb = load(b)
    assert (ia == ib).all() and la.shape == lb.shape, "secuencias distintas"
    pa, pb = logsoftmax(la), logsoftmax(lb)
    kl = (np.exp(pa) * (pa - pb)).sum(-1)
    agree = la.argmax(-1) == lb.argmax(-1)
    d = np.abs(la.astype(np.float64) - lb)
    rel = np.sqrt(((la - lb) ** 2).mean(-1)) / np.sqrt((lb ** 2).mean(-1))
    if verbose:
        for i in range(len(ia)):
            print(f"  pos {i:3d}: argmax {'=' if agree[i] else '!='} "
                  f"max diff {d[i].max():.3g}  rel rms {rel[i]:.2e}  KL {kl[i]:.2e}")
    print(f"argmax equal {agree.sum()}/{len(agree)}; max diff {d.max():.3g}; "
          f"mean rel rms {rel.mean():.2e}; mean KL {kl.mean():.2e}, max {kl.max():.2e}")
    return agree, kl, rel


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], "-q" not in sys.argv)
