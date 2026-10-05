"""Recheck the published finite-array witnesses without loading a checkpoint.

The all-vocabulary upper Lipschitz bound additionally needs the checkpoint;
reachable.py performs that check. This replay proves the lower bounds needed
for refusal, the exact group-allocation threshold, and the programme budget.
"""

import argparse
import heapq
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from audit import geometry, load_lib, mass_lower

ROOT = Path(__file__).resolve().parents[1]


def rational(x):
    return F(float(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="compact_replay.json")
    args = ap.parse_args()
    start = time.perf_counter()
    w = np.load(ROOT / "data/reachable_norm_witness.npz", allow_pickle=False)
    x = [rational(t) for t in w["x"]]
    b = [
        (rational(a) - rational(c)) * rational(g)
        for a, c, g in zip(w["row1"], w["row2"], w["gamma"])
    ]
    d = len(x)
    s = sum(t * t for t in x)
    a = s / d + rational(w["eps"])
    bb = sum(t * t for t in b)
    bx = sum(t * u for t, u in zip(b, x))
    grad2 = (a * a * bb - 2 * a * bx * bx / d + bx * bx * s / (d * d)) / (a * a * a)
    C = F(7, 25)
    op = F(1937, 100)
    assert s >= 948**2 and grad2 >= C * C
    vi = [int(t) for t in w["projection_vector"]]
    image = [sum(rational(t) * v for t, v in zip(row, vi)) for row in w["projection"]]
    assert sum(t * t for t in image) >= op * op * sum(t * t for t in vi)
    # The headwise appendix check uses actual columns of the same stored O.
    for h in range(9):
        block = w["projection"][:, 64 * h : 64 * (h + 1)]
        j = int(np.argmax((block.astype(np.float64) ** 2).sum(axis=0)))
        assert sum(rational(v) ** 2 for v in block[:, j]) >= 9
    snap = np.load(ROOT / "data/last_fp32.npz", allow_pickle=False)
    indices_file = ROOT / "data/clique_indices.npz"
    ids = np.load(indices_file, allow_pickle=False) if indices_file.exists() else None
    lib = load_lib()
    functions = []
    sizes = []
    n = snap["k"].shape[1]
    for h in range(9):
        q = np.ascontiguousarray(snap["q"][h])
        k = np.ascontiguousarray(snap["k"][h // 3])
        v = np.ascontiguousarray(snap["v"][h // 3])
        inds = np.ascontiguousarray(
            ids[f"h{h}"] if ids else np.load(ROOT / "data" / f"local_packing_h{h}.npz")["indices"],
            dtype=np.int64,
        )
        assert len(set(map(int, inds))) == len(inds) and inds.min() >= 0 and inds.max() < n
        ki, vi, qi = geometry(q, k, v)
        assert lib.verify(ki, vi, qi, inds, n, 64, 4 * (1 << 20), 1 << 9, len(inds)) == 1
        L, Z, *_ = mass_lower(q, k)
        masses = sorted((L[int(i)] for i in inds), reverse=True)
        tail = sum(masses)
        curve = [F(tail, 20 * Z) ** 2]
        for m in masses:
            tail -= m
            curve.append(F(tail, 20 * Z) ** 2)
        savings = [curve[j] - curve[j + 1] for j in range(len(curve) - 1)]
        assert all(a >= b >= 0 for a, b in zip(savings, savings[1:]))
        functions.append(curve)
        sizes.append(len(inds))
    target2 = (F(1, 250) / (C * op)) ** 2
    alloc = [0] * 9
    current = sum(f[0] for f in functions)
    heap = [(-(f[0] - f[1]), h) for h, f in enumerate(functions)]
    heapq.heapify(heap)
    checks = {}
    for budget in range(18634):
        if budget in [6451, 18632, 18633]:
            nexts = [
                f[alloc[h]] - f[alloc[h] + 1]
                for h, f in enumerate(functions)
                if alloc[h] + 1 < len(f)
            ]
            lasts = [f[alloc[h] - 1] - f[alloc[h]] for h, f in enumerate(functions) if alloc[h] > 0]
            assert max(nexts) <= min(lasts)
            lam = (max(nexts) + min(lasts)) / 2
            # Exact supporting lines are an independent optimality proof.
            for h, f in enumerate(functions):
                m = alloc[h]
                assert all(y + lam * j >= f[m] + lam * m for j, y in enumerate(f))
            assert (current > target2) == (budget < 18633)
            checks[str(budget)] = {
                "allocation": alloc.copy(),
                "sum_squared_exact": str(current),
                "dual_lambda_exact": str(lam),
                "rejected": current > target2,
            }
        if budget == 18633:
            break
        neg, h = heapq.heappop(heap)
        current += neg
        alloc[h] += 1
        j = alloc[h]
        f = functions[h]
        if j + 1 < len(f):
            heapq.heappush(heap, (-(f[j] - f[j + 1]), h))
    delta = F(1, 10**12) + F(49151, 1 << 64)
    reserve = F(924, 10**6) - 2048 * delta
    assert reserve > 0 and F(1024, 1200000000) < reserve * reserve
    report = {
        "readout_lower": "7/25",
        "radius": "1",
        "hidden_norm_lower": "948",
        "readout_derivative_squared_exact": str(grad2),
        "projection_lower": "1937/100",
        "clique_sizes": sizes,
        "dense_interactions": 9 * n,
        "allocation_checks": checks,
        "sequence_TV_strict_upper": "0.000924",
        "all_vocabulary_upper_checked_here": False,
        "seconds": time.perf_counter() - start,
    }
    (ROOT / "results" / args.output).write_text(json.dumps(report, indent=2))
    print(
        "Checked reached-state lower gain, projection, nine cliques, three dual allocation witnesses, and sequence budget."
    )
    print(
        "Tenfold budget 6451 rejected; 18632 still rejected; 18633 is the first nonrejected lower-bound budget."
    )
    print(f"Total wall time: {report['seconds']:.3f} seconds")


if __name__ == "__main__":
    main()
