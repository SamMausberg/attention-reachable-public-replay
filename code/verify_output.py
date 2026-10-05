"""Independent rational checks of saved admission witnesses and sampler laws."""

import hashlib
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from categorical import bounds, weights
from fastlaw import law

ROOT = Path(__file__).resolve().parents[1]


def main():
    t = time.perf_counter()
    out = {}
    files = [
        "logits_fp32",
        "logits_tail_reference",
        "logits_scores_64",
        "logits_scores_256",
        "logits_time_0",
        "logits_scores_0",
        "logits_bf16_attention",
        "logits_fp64_attention",
    ]
    laws = {}
    for f in files:
        z = np.load(ROOT / "data" / (f + ".npy"))
        laws[f] = weights(z)
        assert laws[f]["floor_resolved"]
        c, r = law(z)
        diff = F(sum(abs(int(a) - b) for a, b in zip(c, laws[f]["counts"])), 2 * (1 << 64))
        assert diff <= F(1, 10**12) + 2 * F(len(z) - 1, 1 << 64)
        out[f] = {
            "fast_sampler_vs_high_precision_TV_exact": str(diff),
            "array_sha256": hashlib.sha256(z.tobytes()).hexdigest(),
        }
    comparisons = {}
    for f in files:
        if f != "logits_tail_reference":
            comparisons[f] = bounds(laws["logits_tail_reference"], laws[f])
    comparisons["bf16_vs_fp32"] = bounds(laws["logits_fp32"], laws["logits_bf16_attention"])
    comparisons["fp64_vs_fp32"] = bounds(laws["logits_fp32"], laws["logits_fp64_attention"])
    proofs = []
    for path in sorted(
        list((ROOT / "data").glob("technical_replay_step*.npz"))
        + list((ROOT / "data").glob("alice_step*.npz"))
    ):
        z = np.load(path)
        p = z["reference"]
        q = z["candidate"]
        try:
            c, r = law(q, p)
        except (ValueError, RuntimeError) as error:
            # The candidate's conservative guard must select the reference count law.
            tag, step = path.stem.rsplit("_step", 1)
            record = json.loads((ROOT / "results" / (tag + ".json")).read_text())["records"][
                int(step)
            ]
            assert not record["accepted"] and record["certificate"]["reason"] == str(error)
            try:
                served, _ = law(p)
            except (ValueError, RuntimeError):
                served = weights(p)["counts"]
            assert list(map(int, z["counts"])) == list(map(int, served))
            high = weights(p)
            distance = F(
                sum(abs(int(a) - int(b)) for a, b in zip(served, high["counts"])), 2 * (1 << 64)
            )
            assert distance <= F(1, 10**12) + 2 * F(len(p) - 1, 1 << 64)
            proofs.append(
                {
                    "snapshot": path.name,
                    "guard": str(error),
                    "accepted": False,
                    "served_reference_counts_match": True,
                    "fast_reference_sampler_vs_independent_TV_exact": str(distance),
                }
            )
            continue
        assert list(map(int, z["counts"])) == list(map(int, c if r["admit"] else law(p)[0]))
        L = weights(q)
        D = 1 << r["delta_grid_bits"]
        C = r["center"]
        delta = [F(float(a)) - F(float(b)) - F(C, D) for a, b in zip(p, q)]
        # Independent 256-bit outward upper second moment from exact rational deltas.
        second = sum(F(u) * d * d for u, d in zip(L["U"], delta)) / L["ZL"]
        ran = F(r["osc_numerator"], r["osc_denominator"])
        kb = second / (2 * (1 - ran))
        fast = F(int(r["kl_numerator"]), int(r["kl_denominator"]))
        assert kb <= fast
        proofs.append(
            {
                "snapshot": path.name,
                "independent_KL_bound_exact": str(kb),
                "fast_KL_bound_exact": str(fast),
                "accepted": r["admit"],
            }
        )
    # This is a worst-case program guarantee, not the recorded-path sum.
    delta = F(1, 10**12) + F(49151, 1 << 64)
    T = 1024
    tv_limit = F(924, 10**6)
    # Prove sqrt(T/(2*600000000))+2*T*delta < .000924 by squaring.
    reserve = tv_limit - 2 * T * delta
    assert reserve > 0 and F(T, 1200000000) < reserve * reserve
    report = {
        "samplers": out,
        "comparisons": comparisons,
        "independent_moment_checks": proofs,
        "sequence_TV_strict_upper": "0.000924",
        "per_token_sampler_TV_upper_exact": str(delta),
        "squared_Pinsker_budget_exact": str(F(T, 1200000000)),
        "seconds": time.perf_counter() - t,
    }
    (ROOT / "results/output_verification.json").write_text(json.dumps(report, indent=2))
    print(
        "CHECKED", len(proofs), "paired snapshots and", len(out), "sampler comparisons", flush=True
    )
    for k, v in comparisons.items():
        print(k, v["integer_sampler_TV"], flush=True)


if __name__ == "__main__":
    main()
