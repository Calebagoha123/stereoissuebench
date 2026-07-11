#!/usr/bin/env python3
"""Generation-variance check (F): are the cue effects sampling noise at T=0.7?

Each cell (model x cue x issue x template) is generated 3x. We decompose the
variance of the per-response liberal score into within-cell (between-generation,
temperature) vs between-cell (condition) components, report the intraclass
correlation, and recompute each cue effect Delta_k per replicate to show the
effects are stable across independent generations rather than an artifact of one
lucky draw.

Reads results/full_3x/bert_eval_*.csv. Writes results/robustness/generation_variance.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, CUE_ORDER, CUE_DISPLAY, ROBUST, load_model


def variance_components(d):
    """One-way ANOVA decomposition with the cell (issue x template x cue) as group."""
    d = d.copy()
    d["cell"] = d["issue_id"].astype(str) + "|" + d["template_id"].astype(str) + "|" + \
        d["cue_family"].astype(str) + "|" + d["cue_group"].astype(str)
    grand = d["y"].mean()
    g = d.groupby("cell")["y"]
    cell_mean = g.transform("mean")
    ss_within = ((d["y"] - cell_mean) ** 2).sum()
    ss_between = ((cell_mean - grand) ** 2).sum()
    ss_total = ((d["y"] - grand) ** 2).sum()
    # ICC(1): between-cell variance share
    k = g.count().mean()
    ms_b = ss_between / (g.ngroups - 1)
    ms_w = ss_within / (len(d) - g.ngroups)
    icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
    return {"var_within_frac": ss_within / ss_total,
            "var_between_frac": ss_between / ss_total, "icc1": icc}


def main():
    recs = []
    per_rep = []
    for m in MODELS:
        d = load_model(m)
        vc = variance_components(d)
        recs.append({"model": m, **vc})
        # per-replicate cue effects
        base_all = d[d.cue_family == "baseline"]
        for fam, grp in CUE_ORDER:
            cue = d[(d.cue_family == fam) & (d.cue_group == grp)]
            if cue.empty:
                continue
            base = (base_all[base_all.template_id.isin(cue.template_id.unique())]
                    if fam.startswith("implicit") else base_all)
            row = {"model": m, "cue_display": CUE_DISPLAY[(fam, grp)]}
            deltas = []
            for r in ["r01", "r02", "r03"]:
                dc = cue[cue.rep == r]["y"].mean() - base[base.rep == r]["y"].mean()
                row[r] = dc
                deltas.append(dc)
            row["rep_sd"] = np.std(deltas, ddof=1)
            row["rep_range"] = max(deltas) - min(deltas)
            per_rep.append(row)

    vcdf = pd.DataFrame(recs)
    prdf = pd.DataFrame(per_rep)
    prdf.to_csv(ROBUST / "generation_variance_per_rep.csv", index=False)
    vcdf.to_csv(ROBUST / "generation_variance.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== Generation variance (F) ===\n")
    print("Variance decomposition of per-response liberal score:")
    for _, r in vcdf.iterrows():
        print(f"  {MODEL_LABEL[r['model']]:>14}:  between-cell={r['var_between_frac']*100:.0f}%  "
              f"within-cell(gen)={r['var_within_frac']*100:.0f}%  ICC(1)={r['icc1']:.3f}")
    print("\nPer-replicate cue effects — SD across the 3 independent generations:")
    print(f"  median rep-to-rep SD of Delta_k: {prdf['rep_sd'].median():.4f}")
    print(f"  max    rep-to-rep SD of Delta_k: {prdf['rep_sd'].max():.4f}")
    big = prdf[prdf[["r01", "r02", "r03"]].abs().max(axis=1) > 0.1]
    print(f"\nLargest effects are stable across reps (|Δ|>0.1 cells, rep range):")
    for _, r in big.sort_values("rep_range", ascending=False).head(8).iterrows():
        print(f"  {r['model']:>5} {r['cue_display']:>20}: "
              f"r01={r['r01']:+.3f} r02={r['r02']:+.3f} r03={r['r03']:+.3f}  range={r['rep_range']:.3f}")
    print(f"\nWrote {ROBUST/'generation_variance.csv'}")


if __name__ == "__main__":
    main()
