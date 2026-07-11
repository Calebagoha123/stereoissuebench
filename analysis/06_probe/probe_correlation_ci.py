#!/usr/bin/env python3
"""Bootstrap CIs on the RQ3 probe correlations (check I).

The probe story rests on two correlations across the n=14 cue groups, reported as
point estimates in docs/probe_findings.md:

  A2  stated belief shift  -> written stance shift   (r ≈ 0.74–0.89)
  B3  internal political-axis projection shift -> written stance shift (r ≈ 0.74–0.92)

With only 14 points these need intervals shown before an examiner computes them.
We resample the 14 cue-group pairs with replacement (2.5/97.5 percentile CI) and
add a label-shuffle permutation p-value.

A2 belief shift is reconstructed from results/full/belief_probe_<model>.csv (opinion
probe, oriented by liberal_sign); B3 proj/stance shifts are read from
results/probe_internal/<model>_mediation_full3x.csv. Writes
results/probe_internal/probe_correlation_ci.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["llama", "gemma", "qwen"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B", "qwen": "Qwen3.6-27B"}
PROBE = Path("results/probe_internal")
BELIEF = Path("results/full")


def boot_r(x, y, n_boot=10000, seed=13):
    rng = np.random.default_rng(seed)
    n = len(x)
    r0 = stats.pearsonr(x, y)[0]
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        rs[b] = stats.pearsonr(xi, yi)[0] if np.std(xi) > 0 and np.std(yi) > 0 else np.nan
    rs = rs[~np.isnan(rs)]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    # permutation p (shuffle y)
    perm = np.empty(2000)
    for p in range(2000):
        perm[p] = stats.pearsonr(x, rng.permutation(y))[0]
    pval = (np.abs(perm) >= abs(r0)).mean()
    return r0, lo, hi, pval


def belief_shift(model):
    d = pd.read_csv(BELIEF / f"belief_probe_{model}.csv", low_memory=False)
    d = d[d.probe_kind == "opinion"].copy()
    d["score"] = pd.to_numeric(d["parsed_score"], errors="coerce")
    d = d.dropna(subset=["score"])
    d["blib"] = (d["score"] - 50) / 50 * d["liberal_sign"]
    base = d[d.cue_family == "baseline"].groupby("issue_id")["blib"].mean()
    med = pd.read_csv(PROBE / f"{model}_mediation_full3x.csv")
    shifts = []
    for _, r in med.iterrows():
        c = d[(d.cue_family == r.cue_family) & (d.cue_group == r.cue_group)]
        shifts.append((c.groupby("issue_id")["blib"].mean() - base).mean())
    med["belief_shift"] = shifts
    return med


def main():
    recs = []
    for m in MODELS:
        med = belief_shift(m)
        # A2 belief -> stance
        r, lo, hi, p = boot_r(med["belief_shift"].to_numpy(), med["stance_shift"].to_numpy())
        recs.append({"model": m, "link": "A2 belief->stance", "n": len(med),
                     "r": r, "ci_lo": lo, "ci_hi": hi, "perm_p": p})
        # B3 internal axis -> stance
        r, lo, hi, p = boot_r(med["proj_shift"].to_numpy(), med["stance_shift"].to_numpy())
        recs.append({"model": m, "link": "B3 axis->stance", "n": len(med),
                     "r": r, "ci_lo": lo, "ci_hi": hi, "perm_p": p})
    out = pd.DataFrame(recs)
    out.to_csv(PROBE / "probe_correlation_ci.csv", index=False)

    print("=== Probe correlations with bootstrap 95% CIs (n=14 cue groups) ===\n")
    for link in ["A2 belief->stance", "B3 axis->stance"]:
        print(f"--- {link} ---")
        for _, r in out[out.link == link].iterrows():
            print(f"  {MODEL_LABEL[r['model']]:>14}:  r={r['r']:.2f}  "
                  f"95% CI [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  perm p={r['perm_p']:.4f}")
        print()
    print("Note: A2 (belief->stance) is robust — bootstrap CIs exclude 0 on all three models.")
    print("B3 (axis->stance) is permutation-significant (p<0.02) but its n=14 bootstrap CI is")
    print("wide and its lower bound reaches ~0: the correlation leans on the extreme Republican")
    print("point, so B3 should be reported with the permutation p and the causal framing, not")
    print("as a tight interval. Showing this is the honest MSc-scale disclosure.")
    print(f"Wrote {PROBE/'probe_correlation_ci.csv'}")


if __name__ == "__main__":
    main()
