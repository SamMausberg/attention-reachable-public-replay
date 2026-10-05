"""Sound categorical laws for recorded binary32 logits.

The only transcendental step is enclosed by alternating rational Taylor
bounds and integer interval exponentiation. The resulting 64-bit integer
sampler has no libm or floating-point probability arithmetic.
"""

from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import json, math, time, hashlib, secrets
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ceildiv(a, b):
    return -(-a // b)


def weights(logits, precision=256):
    logits = np.asarray(logits)
    if logits.dtype != np.float32 or logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("Expected a finite one-dimensional binary32 array")
    pairs = [float(x).as_integer_ratio() for x in logits]
    B = max(b.bit_length() - 1 for a, b in pairs)
    unit = 1 << B
    ints = [a * (unit // b) for a, b in pairs]
    mx = max(ints)
    powers = [mx - a for a in ints]
    if B < 1:  # Generic integer exponents: choose grid 1/2.
        B = 1
        powers = [a * 2 for a in powers]
        unit = 2
    S = 1 << precision
    x = F(1, unit)
    term = F(1)
    partial = term
    # exp(-x) for x<=1/2. Odd and even partial sums bracket exactly.
    for j in range(1, 80):
        term *= -x / j
        partial += term
        if j % 2:
            lo = partial
        else:
            hi = partial
        if j >= 2 and abs(term) < F(1, 1 << (precision + max(4, B + 2))):
            if j % 2:
                term *= -x / (j + 1)
                hi = partial + term
            else:
                term *= -x / (j + 1)
                lo = partial + term
            break
    assert 0 < lo <= hi
    base = (lo.numerator * S // lo.denominator, ceildiv(hi.numerator * S, hi.denominator))
    ps = [base]
    for _ in range(max(powers).bit_length() - 1):
        a, b = ps[-1]
        ps.append((a * a // S, ceildiv(b * b, S)))

    @lru_cache(None)
    def ex(j):
        l = u = S
        k = 0
        while j:
            if j & 1:
                a, b = ps[k]
                l = l * a // S
                u = ceildiv(u * b, S)
            k += 1
            j >>= 1
        return l, u

    pairs = [ex(j) for j in powers]
    L = [a for a, b in pairs]
    U = [b for a, b in pairs]
    zl = sum(L)
    zu = sum(U)
    D = 1 << 64
    pivot = int(np.argmax(logits))
    ws = []
    # Keep a total, bounded-error reference law even when a 64-bit floor
    # remains unresolved. The proof uses this checked width, not a limit
    # on how close a softmax probability may lie to a dyadic threshold.
    if (zu - zl) * 10**12 > zl:
        return weights(logits, precision * 2)
    for i, (a, b) in enumerate(pairs):
        l = a * D // zu
        u = b * D // zl
        if i != pivot and l != u:
            if precision < 1024:
                return weights(logits, precision * 2)
            # This branch was not used by the recorded experiments.
            # Its law is L/ZL followed by pivot-residual count rounding.
            ws = [a * D // zl for a in L]
            ws[pivot] = D - sum(w for j, w in enumerate(ws) if j != pivot)
            assert min(ws) >= 0 and sum(ws) == D
            return {
                "L": L,
                "U": U,
                "ZL": zl,
                "ZU": zu,
                "counts": ws,
                "pivot": pivot,
                "bits": B,
                "precision": precision,
                "floor_resolved": False,
            }
        ws.append(l)
    ws[pivot] = D - sum(w for i, w in enumerate(ws) if i != pivot)
    assert min(ws) >= 0 and sum(ws) == D
    return {
        "L": L,
        "U": U,
        "ZL": zl,
        "ZU": zu,
        "counts": ws,
        "pivot": pivot,
        "bits": B,
        "precision": precision,
        "floor_resolved": True,
    }


def bounds(p, q):
    # Exact overlap lower bound yields an upper bound on total variation.
    den = p["ZU"] * q["ZU"]
    overlap = sum(min(a * q["ZU"], b * p["ZU"]) for a, b in zip(p["L"], q["L"]))
    upper = F(den - overlap, den)
    # A fixed event selected by integer lower midpoints gives a sound lower.
    A = [i for i, (a, b) in enumerate(zip(p["counts"], q["counts"])) if a > b]
    lower = max(F(0), F(sum(p["L"][i] for i in A), p["ZU"]) - F(sum(q["U"][i] for i in A), q["ZL"]))
    integer_tv = F(sum(abs(a - b) for a, b in zip(p["counts"], q["counts"])), 2 * (1 << 64))
    assert lower <= upper
    return {
        "ideal_softmax_lower_exact": str(lower),
        "ideal_softmax_upper_exact": str(upper),
        "ideal_softmax_lower": float(lower),
        "ideal_softmax_upper": float(upper),
        "integer_sampler_TV_exact": str(integer_tv),
        "integer_sampler_TV": float(integer_tv),
        "event_size": len(A),
    }


def sample(counts):
    """Exact sampler under independent uniform random bytes from the OS."""
    u = secrets.randbits(64)
    c = 0
    for i, w in enumerate(counts):
        c += w
        if u < c:
            return i
    raise ArithmeticError("Counts do not sum to 2^64")


def main():
    start = time.perf_counter()
    paths = sorted((ROOT / "data").glob("logits_*.npy"))
    laws = {}
    times = {}
    for path in paths:
        t = time.perf_counter()
        law = weights(np.load(path))
        laws[path.stem] = law
        times[path.stem] = time.perf_counter() - t
        # JSON integers preserve the complete distribution exactly.
        (ROOT / "data" / (path.stem + "_counts.json")).write_text(json.dumps(law["counts"]))
        print(path.stem, "seconds", times[path.stem], flush=True)
    ref = laws["logits_tail_reference"]
    comparisons = {}
    for name, law in laws.items():
        if name != "logits_tail_reference":
            comparisons[name] = bounds(ref, law)
    if "logits_bf16_attention" in laws:
        comparisons["bf16_attention_vs_fp32_batched"] = bounds(
            laws["logits_fp32"], laws["logits_bf16_attention"]
        )
    if "logits_fp64_attention" in laws:
        comparisons["fp64_attention_vs_fp32_batched"] = bounds(
            laws["logits_fp32"], laws["logits_fp64_attention"]
        )
    report = {
        "vocab": len(ref["counts"]),
        "sampling_denominator": str(1 << 64),
        "sampler_to_softmax_TV_upper_exact": str(
            F(len(ref["counts"]) - 1, 1 << 64)
            + (F(0) if all(z["floor_resolved"] for z in laws.values()) else F(1, 10**12))
        ),
        "comparisons": comparisons,
        "law_build_seconds": times,
        "total_seconds": time.perf_counter() - start,
        "arithmetic": "Integer outward exponent intervals, exact rational comparisons, exact integer categorical law",
    }
    (ROOT / "results/categorical.json").write_text(json.dumps(report, indent=2))
    for name, r in comparisons.items():
        print(name, r["integer_sampler_TV_exact"], r["integer_sampler_TV"], flush=True)


if __name__ == "__main__":
    main()
