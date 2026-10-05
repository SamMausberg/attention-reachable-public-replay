"""Minimal read-only GGUF v2/v3 reader for F32/F16/Q8_0 tensors.
The supported subset fails closed on every other tensor encoding.
"""

import struct, math, hashlib
from pathlib import Path
import numpy as np


class GGUF:
    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        self.pos = 0
        assert self.take(4) == b"GGUF"
        self.version = self.scalar("I")
        assert self.version in (2, 3)
        nt = self.scalar("Q")
        nk = self.scalar("Q")
        self.meta = {}
        for _ in range(nk):
            key = self.string()
            typ = self.scalar("I")
            self.meta[key] = self.value(typ)
        self.tensors = {}
        for _ in range(nt):
            name = self.string()
            nd = self.scalar("I")
            shape = [self.scalar("Q") for _ in range(nd)]
            typ = self.scalar("I")
            off = self.scalar("Q")
            self.tensors[name] = (shape, typ, off)
        align = self.meta.get("general.alignment", 32)
        self.start = (self.pos + align - 1) // align * align

    def take(self, n):
        if self.pos + n > len(self.raw):
            raise ValueError("truncated file")
        b = self.raw[self.pos : self.pos + n]
        self.pos += n
        return b

    def scalar(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.take(self.scalar("Q")).decode("utf8")

    def value(self, t):
        fmts = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }
        if t in fmts:
            return self.scalar(fmts[t])
        if t == 8:
            return self.string()
        if t == 9:
            typ = self.scalar("I")
            n = self.scalar("Q")
            return [self.value(typ) for _ in range(n)]
        raise ValueError(f"unsupported metadata type {t}")

    def tensor(self, name):
        shape, typ, off = self.tensors[name]
        n = math.prod(shape)
        off += self.start
        if typ in (0, 1):
            a = np.frombuffer(
                self.raw, dtype="<f4" if typ == 0 else "<f2", count=n, offset=off
            ).astype(np.float32, copy=True)
        elif typ == 8:
            if n % 32:
                raise ValueError("unaligned Q8_0 tensor")
            dt = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
            b = np.frombuffer(self.raw, dtype=dt, count=n // 32, offset=off)
            a = (b["d"].astype(np.float32)[:, None] * b["q"].astype(np.float32)).reshape(-1)
        else:
            raise ValueError(f"unsupported tensor type {typ}")
        return a.reshape(tuple(reversed(shape)))


if __name__ == "__main__":
    import sys, collections

    g = GGUF(sys.argv[1])
    print(g.sha256)
    for k, v in g.meta.items():
        if not isinstance(v, list):
            print(k, repr(v))
        else:
            print(k, "array", len(v), repr(v[:5]))
    print("types", collections.Counter(v[1] for v in g.tensors.values()))
    print(list(g.tensors.items())[:15])
