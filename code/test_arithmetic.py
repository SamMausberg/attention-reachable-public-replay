"""Arithmetic edge controls; tests complement, and do not replace, the proofs."""

import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from categorical import weights
from fastlaw import law

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(103)
    cases = 0
    moment = 0
    precision_guards = 0
    for n in [2, 3, 7, 32, 257]:
        zs = [np.zeros(n, np.float32), np.linspace(-30, 0, n, dtype=np.float32)]
        zs += [rng.normal(0, 3, n).astype(np.float32) for _ in range(5)]
        for z in zs:
            exact = weights(z)
            c, r = law(z)
            assert sum(map(int, c)) == 1 << 64 and sum(exact["counts"]) == 1 << 64
            dist = F(sum(abs(int(a) - b) for a, b in zip(c, exact["counts"])), 2 * (1 << 64))
            assert dist <= F(1, 10**12) + F(2 * (n - 1), 1 << 64)
            p = (z + rng.normal(0, 0.0001, n).astype(np.float32)).astype(np.float32)
            try:
                _, rr = law(z, p)
            except ValueError as e:
                assert "precision guard" in str(e).lower()
                precision_guards += 1
                cases += 1
                continue
            den = 1 << rr["delta_grid_bits"]
            center = F(rr["center"], den)
            dd = [F(float(a)) - F(float(b)) - center for a, b in zip(p, z)]
            rrange = F(rr["osc_numerator"], rr["osc_denominator"])
            if rrange < 1:
                independent = (
                    sum(F(u) * d * d for u, d in zip(exact["U"], dd))
                    / exact["ZL"]
                    / (2 * (1 - rrange))
                )
                fast = F(int(rr["kl_numerator"]), int(rr["kl_denominator"]))
                assert independent <= fast
                assert rr["admit"] == (fast <= F(1, 600000000))
                moment += 1
            cases += 1
    # Point-mass-to-count precision and uniform dyadic floors.
    c = weights(np.array([1000.0, 0.0], np.float32))["counts"]
    assert c == [1 << 64, 0]
    c = weights(np.array([0.0, 0.0], np.float32))["counts"]
    assert c == [1 << 63, 1 << 63]
    for z in [
        np.array([np.nan, 0], np.float32),
        np.array([np.inf, 0], np.float32),
        np.array([np.nextafter(np.float32(0), np.float32(1)), 1], np.float32),
    ]:
        try:
            law(z)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid or guarded input was admitted")
    # A common logit shift has zero range and zero variance after centring.
    _, r = law(np.array([0.0, 1.0, 2.0], np.float32), np.array([0.5, 1.5, 2.5], np.float32))
    assert r["admit"] and int(r["kl_numerator"]) == 0
    report = {
        "sampler_cases": cases,
        "independent_moment_checks": moment,
        "expected_precision_guards": precision_guards,
        "edge_controls_passed": True,
        "seconds": time.perf_counter() - start,
    }
    (ROOT / "results/arithmetic_tests.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
