import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = ROOT / "results"
fig = ROOT / "paper/figures"
fig.mkdir(exist_ok=True)


def read(n):
    return json.loads((res / (n + ".json")).read_text())


a = read("reachable_packing")
t = read("technical_replay")
s = read("alice")
v = read("output_verification")
vals = {
    "TechAccept": t["accepted"],
    "LitAccept": s["accepted"],
    "TechFallback": t["fallbacks"],
    "LitFallback": s["fallbacks"],
    "TechPrefill": f"{t['prefill_seconds']:.3f}",
    "TechDecode": f"{t['decode_all_overheads_seconds']:.3f}",
    "TechTotal": f"{t['total_with_load_seconds']:.3f}",
    "LitTotal": f"{s['total_with_load_seconds']:.3f}",
    "TechRSS": f"{t['peak_rss_KiB'] / 1024:.1f}",
    "PackingSeconds": f"{a['total_check_seconds']:.3f}",
}
for name, k in [
    ("TechLower", "lower_layers_and_updates"),
    ("TechRef", "reference_terminal"),
    ("TechCand", "candidate_terminal"),
    ("TechCheck", "law_and_check"),
]:
    vals[name] = f"{t['phase_seconds'][k]:.3f}"
for name, n in [("TechBase", "baseline_technical_replay"), ("LitBase", "baseline_alice")]:
    p = res / (n + ".json")
    vals[name] = f"{read(n)['total_with_load_seconds']:.3f}" if p.exists() else "PENDING"
(ROOT / "paper/numbers.tex").write_text(
    "\n".join("\\newcommand{\\" + k + "}{" + str(x) + "}" for k, x in vals.items()) + "\n"
)
(fig / "joint.dat").write_text(
    "percent osc\n"
    + "\n".join(
        f"{100 * p['groups'] / 64512:.12g} {p['osc_lower']:.12g}" for p in a["joint"]["curve"]
    )
)
comp = v["comparisons"]
fs = [
    "logits_time_0",
    "logits_scores_0",
    "logits_scores_64",
    "logits_scores_256",
    "bf16_vs_fp32",
    "fp64_vs_fp32",
]
(fig / "tv.dat").write_text(
    "row tv\n"
    + "\n".join(f"{j + 1} {comp[x]['integer_sampler_TV']:.15g}" for j, x in enumerate(fs))
)
reject = []
for tag, data in [("technical", t), ("alice", s)]:
    rows = []
    for r in data["records"]:
        c = r["certificate"]
        den = int(c.get("kl_denominator", 0))
        num = int(c.get("kl_numerator", 0))
        if den and num:
            kval = float(F(num, den))
            rows.append(f"{r['step']} {kval:.14g}")
            if not r["accepted"]:
                reject.append(f"{r['step']} {kval:.14g}")
    (fig / (tag + ".dat")).write_text("step kl\n" + "\n".join(rows))
(fig / "rejected.dat").write_text("step kl\n" + "\n".join(reject))
# Verify that the rounded decimal interval printed in the manuscript encloses
# both the exact finite sampler law and the independently enclosed softmax law.
c = comp["logits_scores_256"]
lo = F("1.14762374e-5")
hi = F("1.14762375e-5")
assert lo <= F(c["ideal_softmax_lower_exact"]) <= F(c["ideal_softmax_upper_exact"]) <= hi
assert lo <= F(c["integer_sampler_TV_exact"]) <= hi
print(vals)
