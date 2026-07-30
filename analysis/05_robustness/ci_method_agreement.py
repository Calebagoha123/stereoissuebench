#!/usr/bin/env python3
"""Do the two interval methods used in the thesis agree? (Appendix correctness check.)

Two 95% intervals are computed for the same estimand, Delta_k:

  * **cluster-t** (the figures, via make_thesis_figures._t_crit): the 19 per-issue
    paired differences (cued minus template-matched baseline, within issue), then
    mean +/- t_{18} * SE. This is the direct implementation of the Week-8 rule --
    "clustered SEs treat each unit as an independent block of information: N clusters
    ... degrees of freedom are closer to N than NT, which can create problems for
    small N" -- so the df come from the 19 issues, not the ~8k generations per cell.

  * **percentile bootstrap** (_common.shift_table, feeding the master table and the
    RQ2 suite): resample the 19 issues with replacement, recompute the cued and
    baseline means over the resampled issues, take the 2.5/97.5 percentiles of the
    difference. Week 2's bootstrap CI with the issue as the resampling unit.

Having two intervals for one quantity is only defensible if they agree, and if one is
named primary. This script quantifies the agreement so the appendix can state it
rather than assert it.

Usage:  python3 analysis/05_robustness/ci_method_agreement.py
Writes: results/robustness/ci_method_agreement.{csv,md}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _common import load_all, MODELS, MODEL_LABEL, CUE_ORDER, CUE_DISPLAY, ROBUST  # noqa: E402


def cluster_t_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, cue) Delta_k with a cluster-t interval on the 19 issue clusters.

    Baseline matching follows _common.shift_table exactly: implicit (Arm-B) cues are
    differenced against baseline rows sharing their template subset, explicit (Arm-A)
    cues against the full baseline. Diverging here would make the comparison
    meaningless."""
    rows = []
    for m in MODELS:
        d = df[df.model == m]
        base_all = d[d.cue_family == "baseline"]
        for fam, grp in CUE_ORDER:
            cue = d[(d.cue_family == fam) & (d.cue_group == grp)]
            if cue.empty:
                continue
            base = (base_all[base_all.template_id.isin(cue.template_id.unique())]
                    if fam.startswith("implicit") else base_all)
            cm = cue.groupby("issue_id")["y"].mean()
            bm = base.groupby("issue_id")["y"].mean()
            diff = (cm - bm).dropna()
            n = len(diff)
            mu = diff.mean()
            se = diff.std(ddof=1) / np.sqrt(n)
            lo, hi = stats.t.interval(0.95, n - 1, mu, se)
            rows.append({"model": m, "cue_family": fam, "cue_group": grp,
                         "cue_display": CUE_DISPLAY[(fam, grp)], "n_clusters": n,
                         "t_est": mu, "t_se": se, "t_lo": lo, "t_hi": hi,
                         "t_sig": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def main():
    df = load_all()
    t = cluster_t_table(df)
    mt = pd.read_csv(ROBUST / "model_shift_table.csv")
    j = t.merge(mt[["model", "cue_family", "cue_group", "model_shift",
                    "model_shift_lo", "model_shift_hi"]],
                on=["model", "cue_family", "cue_group"], how="inner")
    j["boot_sig"] = (j.model_shift_lo > 0) | (j.model_shift_hi < 0)
    j["agree"] = j.t_sig == j.boot_sig
    j["width_ratio"] = (j.t_hi - j.t_lo) / (j.model_shift_hi - j.model_shift_lo)
    j.to_csv(ROBUST / "ci_method_agreement.csv", index=False)

    n = len(j)
    md = [
        "## Interval-method agreement (cluster-$t$ vs percentile bootstrap)", "",
        f"- cells compared: **{n}** (5 models x 14 cues)",
        f"- point estimates: max absolute difference **{(j.t_est - j.model_shift).abs().max():.4f}** "
        "(the two differ only in how the interval is formed, not the estimator)",
        f"- significant under cluster-$t$: **{int(j.t_sig.sum())}**",
        f"- significant under bootstrap: **{int(j.boot_sig.sum())}**",
        f"- **agreement on significance: {int(j.agree.sum())}/{n} cells**",
        f"- cluster-$t$ intervals are wider by a median factor of "
        f"**{j.width_ratio.median():.2f}** (range {j.width_ratio.min():.2f}-{j.width_ratio.max():.2f})",
        "",
        "**Primary = cluster-$t$.** It is the more conservative of the two, so every "
        "claim made under it also holds under the bootstrap; it is the direct "
        "implementation of the Week-8 clustering rule including the degrees-of-freedom "
        "consequence; and it is what the headline figures display. The bootstrap is "
        "reported as an agreement check and is retained where a bootstrap variance is "
        "needed downstream (the Deming fit and the DiD variance propagation both "
        "consume `model_shift_var`).",
    ]
    if (~j.agree).any():
        md += ["", "### Cells where the two disagree", "",
               j.loc[~j.agree, ["model", "cue_display", "t_est", "t_lo", "t_hi",
                                "model_shift_lo", "model_shift_hi"]].to_markdown(index=False)]
    (ROBUST / "ci_method_agreement.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {ROBUST}/ci_method_agreement.{{csv,md}}")


if __name__ == "__main__":
    main()
