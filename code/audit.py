"""Exact-arithmetic refusal certificates for value-aware box summaries.

Floating-point forward evaluation only supplies immutable binary32 snapshots.
Every reported lower bound is subsequently checked using integer arithmetic.
The model executor itself is not formally verified by this program.
"""

from __future__ import annotations

import ctypes
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path

import numpy as np

EXP_BITS = 96
EXP_SCALE = 1 << EXP_BITS
GRID = 1024
# Alternating Taylor series at -1/1024 gives an exact rational enclosure.
x = F(1, GRID)
term = F(1)
partial = F(1)
for j in range(1, 22):
    term *= -x / j
    partial += term
    if j == 20:
        base_hi = partial
base_lo = partial
BASE_LO = (base_lo.numerator * EXP_SCALE) // base_lo.denominator
BASE_HI = -((-base_hi.numerator * EXP_SCALE) // base_hi.denominator)


@lru_cache(maxsize=None)
def exp_grid(j: int):
    """Integer L,U with L/2^96 <= exp(-j/1024) <= U/2^96."""
    if j < 0:
        raise ValueError("negative exponent index")
    lo = hi = EXP_SCALE
    a, b = BASE_LO, BASE_HI
    while j:
        if j & 1:
            lo = lo * a // EXP_SCALE
            hi = (hi * b + EXP_SCALE - 1) // EXP_SCALE
        a = a * a // EXP_SCALE
        b = (b * b + EXP_SCALE - 1) // EXP_SCALE
        j >>= 1
    return lo, hi


def floor_grid(a, bits):
    a = np.asarray(a)
    if a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError("finite binary32 inputs required")
    # Conversion binary32 -> binary64 and scaling by 2^bits are exact here.
    if np.max(np.abs(a), initial=0) > 2**20:
        raise ValueError("input magnitude guard")
    return np.floor(a.astype(np.float64) * (1 << bits)).astype(np.int64)


def score_intervals(q, k):
    bits = 20
    A = 1 << bits
    Q = floor_grid(q, bits)
    K = floor_grid(k, bits)
    d = len(q)
    # This guard is calculated as a Python integer, before any int64 products.
    bound = d * int(np.abs(Q).max()) * int(np.abs(K).max()) + d * (
        int(np.abs(Q).max()) + int(np.abs(K).max()) + 1
    )
    if bound >= 2**62:
        raise ValueError("score dot overflow guard")
    dot = K @ Q
    lo = dot + np.minimum(Q, 0).sum() + np.minimum(K, 0).sum(axis=1)
    hi = dot + np.maximum(Q, 0).sum() + np.maximum(K, 0).sum(axis=1) + d
    if d != 64:
        raise ValueError("this checked implementation uses exact sqrt(64)=8")
    den = 8 * A * A
    return lo, hi, den


def mass_lower(q, k):
    low, high, den = score_intervals(q, k)
    c = int(high.max())
    lo = []
    hi = []
    for l, u in zip(low, high):
        lower_index = ((c - int(l)) * GRID + den - 1) // den
        upper_index = ((c - int(u)) * GRID) // den
        a, _ = exp_grid(lower_index)
        _, b = exp_grid(upper_index)
        lo.append(a)
        hi.append(b)
    return lo, sum(hi), low, high, den


def load_lib():
    lib = ctypes.CDLL(str(Path(__file__).with_name("separation.so")))
    ptr = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    lib.greedy.argtypes = [
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ptr,
    ]
    lib.greedy.restype = ctypes.c_int
    lib.verify.argtypes = [
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.verify.restype = ctypes.c_int
    return lib


def geometry(q, k, v):
    K = floor_grid(k, 10)
    V = floor_grid(v, 10)
    Q = floor_grid(np.abs(q), 10)
    n, d = K.shape
    bound = d * int(Q.max()) * (int(K.max()) - int(K.min()) + 2)
    if bound >= 2**62:
        raise ValueError("geometry overflow guard")
    if int(V.max()) - int(V.min()) >= 2**62:
        raise ValueError("value overflow guard")
    return np.ascontiguousarray(K), np.ascontiguousarray(V), np.ascontiguousarray(Q)


def groups_required(lower, Z, indices, tau, c):
    # At most M clique vertices can occupy good groups. Every other vertex
    # contributes its exact lower mass to bad groups, whose mass is <= tau/c.
    values = sorted(lower[int(i)] for i in indices)
    removed = 0
    s = 0
    budget = tau / c
    for val in values:
        if (s + val) * budget.denominator <= budget.numerator * Z:
            s += val
            removed += 1
        else:
            break
    return len(values) - removed
