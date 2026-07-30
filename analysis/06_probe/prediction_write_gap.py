#!/usr/bin/env python3
"""Inference for the predicted-vs-written figure (§4.4, fig:belief-stance).

The figure is descriptive: one point per cue group x model, so 15-20 points per
panel. That aggregation is a plotting choice, not the design -- the *pairing* lives at
the issue level, because every cue x issue cell has both an elicited opinion prediction
and a written stance for the same issue. This script does the inference at that level
and emits the appendix table.

Two distinct claims, deliberately kept apart because the data supports them unequally:

  (1) UNDER-WRITING, within a cue type. Paired on issue: |predicted shift| minus
      |written shift|, averaged within issue and then across the 19 issues, with a
      t-interval on the 19 issue clusters. Strongly supported in every panel.

  (2) THE TRANSMISSION GRADIENT, between cue types. The per-panel slope of written on
      predicted (the beta quoted in the figure caption), with a bootstrap-over-issues
      CI, plus the party-minus-state contrast. This is the "read but not written"
      claim; with only 19 clusters it is far weaker than (1), and the point of
      computing it is to find out whether it survives at all.

Caveats that the numbers do not remove, and that the appendix note repeats:

  * x is a continuous 0-100 rescaling, y is a mean over three-category stance labels,
    so part of every gap in (1) is a scale artifact rather than hedging. The
    discretisation check (Appendix, robustness) is what speaks to that; this script
    only attaches uncertainty to the gap as measured.
  * |.| before averaging inflates quantities whose true value is near zero, since the
    absolute value of noise is positive. This matters for the name panel, where both
    sides are small: treat its gap as an upper bound. Party and state are far too large
    to be affected.
  * The 15-20 points a slope is fitted on are not independent (the same cue groups
    recur across models), so the slope CI is a statement about issue sampling only.

Usage:  python3 analysis/06_probe/prediction_write_gap.py [--boot 10000] [--seed 20260730]
Writes: results/probe_internal/prediction_write_gap.{csv,md,tex}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
# The written side must come from the shared loader, not from a local re-read: it
# carries the classifier-of-record switch (SCORE_COL = luna_liberal_disc; the
# bert_eval_* files are the superseded DeBERTa labels) and, critically, the Arm-B
# template-matched baseline. Arm-B cues (rotated names and states) were generated on a
# ~35-template subset, so differencing them against the full 145-template baseline
# would confound the cue with template composition -- in exactly the two panels this
# analysis is about.
from _common import MODELS, SCORE_COL, EVAL_PREFIX, load_model  # noqa: E402

BELIEF = Path("results/full")
OUT = Path("results/probe_internal")
# panel order = FOREST_BANDS order (by decreasing directness), as in the figure
FAMS = ["explicit_political", "explicit_demographic",
        "implicit_political", "implicit_demographic"]
FAM_LABEL = {"explicit_political": "Party label",
             "explicit_demographic": r"Race $\times$ gender",
             "implicit_political": "State",
             "implicit_demographic": "Name"}
FAM_PLAIN = {"explicit_political": "Party label", "explicit_demographic": "Race x gender",
             "implicit_political": "State", "implicit_demographic": "Name"}
def issue_level_pairs() -> pd.DataFrame:
    """One row per model x cue group x issue: predicted shift and written shift.

    Both shifts are differenced against the *same issue's* no-cue baseline, so issue
    difficulty cancels -- the same estimator as \\widehat{\\Delta}_k in RQ1. The written
    baseline follows _common.shift_table: implicit (Arm-B) cues are differenced against
    only those baseline rows sharing their template subset, explicit (Arm-A) cues
    against the full baseline."""
    out = []
    for m in MODELS:
        w = load_model(m)
        wbase_all = w[w.cue_family == "baseline"]
        wrows = []
        for (fam, grp), cue in w[w.cue_family != "baseline"].groupby(["cue_family", "cue_group"]):
            base = (wbase_all[wbase_all.template_id.isin(cue.template_id.unique())]
                    if fam.startswith("implicit") else wbase_all)
            bm = base.groupby("issue_id")["y"].mean()
            cm = cue.groupby("issue_id")["y"].mean()
            wrows.append(pd.DataFrame({"cue_family": fam, "cue_group": grp,
                                       "issue_id": cm.index,
                                       "w_shift": cm.values - cm.index.map(bm).values}))
        wc = pd.concat(wrows, ignore_index=True)

        p = pd.read_csv(BELIEF / f"belief_probe_{m}.csv", low_memory=False)
        p = p[p.probe_kind == "opinion"].copy()
        p["s"] = pd.to_numeric(p["parsed_score"], errors="coerce")
        p = p.dropna(subset=["s"])
        # 0-100 -> [-1, 1], then oriented so positive = liberal on that issue
        p["b"] = (p["s"] - 50) / 50 * p["liberal_sign"]
        pbase = p[p.cue_family == "baseline"].groupby("issue_id")["b"].mean()
        pc = (p[p.cue_family != "baseline"]
              .groupby(["cue_family", "cue_group", "issue_id"])["b"].mean().reset_index())
        pc["p_shift"] = pc["b"] - pc.issue_id.map(pbase)

        j = wc.merge(pc[["cue_family", "cue_group", "issue_id", "p_shift"]],
                     on=["cue_family", "cue_group", "issue_id"])
        j["model"] = m
        out.append(j)
    return pd.concat(out, ignore_index=True)


def gap_test(d: pd.DataFrame, fam: str) -> dict:
    """Claim (1): paired |predicted| - |written|, clustered on issue (n = 19, df = 18)."""
    g = d[d.cue_family == fam].copy()
    g["gap"] = g.p_shift.abs() - g.w_shift.abs()
    per_issue = g.groupby("issue_id")["gap"].mean()      # cluster: one value per issue
    n = len(per_issue)
    mu = per_issue.mean()
    se = per_issue.std(ddof=1) / np.sqrt(n)
    t = mu / se
    lo, hi = stats.t.interval(0.95, n - 1, mu, se)
    # The two magnitudes separately, each with its own issue-clustered interval. These
    # are what separate the state panel from the name panel: the slope does not (see the
    # contrasts), but the *prediction* side does -- states draw label-sized predictions
    # and name-sized writing, which is the dissociation the section is about.
    def _ci(series):
        pi = g.groupby("issue_id")[series].apply(lambda s: s.abs().mean())
        m_, s_ = pi.mean(), pi.std(ddof=1) / np.sqrt(len(pi))
        return (m_,) + stats.t.interval(0.95, len(pi) - 1, m_, s_)
    ap_, ap_lo, ap_hi = _ci("p_shift")
    aw_, aw_lo, aw_hi = _ci("w_shift")
    return dict(n_clusters=n, abs_pred=ap_, abs_pred_lo=ap_lo, abs_pred_hi=ap_hi,
                abs_writ=aw_, abs_writ_lo=aw_lo, abs_writ_hi=aw_hi,
                gap=mu, gap_lo=lo, gap_hi=hi, t=t, p=2 * stats.t.sf(abs(t), n - 1))


def _panel_slope(g: pd.DataFrame, issues: np.ndarray | None = None) -> float:
    """Slope of written on predicted for one panel, re-aggregated from issue level.

    Aggregating to cue group x model *inside* the bootstrap keeps the estimand identical
    to the descriptive beta while letting the resampling act on issues, which is the
    level the clustering is at."""
    if issues is not None:
        # index-based take so a repeated issue contributes repeatedly
        g = pd.concat([g[g.issue_id == i] for i in issues], ignore_index=True)
    agg = g.groupby(["model", "cue_group"])[["p_shift", "w_shift"]].mean()
    if len(agg) < 3 or agg.p_shift.std() == 0:
        return np.nan
    return stats.linregress(agg.p_shift, agg.w_shift).slope


def slope_boot(d: pd.DataFrame, n_boot: int, rng: np.random.Generator):
    """Claim (2): per-panel slope + bootstrap-over-issues CI, and the party-state contrast."""
    issues = np.array(sorted(d.issue_id.unique()))
    panels = {f: d[d.cue_family == f].copy() for f in FAMS}
    point = {f: _panel_slope(g) for f, g in panels.items()}
    draws = {f: np.empty(n_boot) for f in FAMS}
    for b in range(n_boot):
        samp = rng.choice(issues, size=len(issues), replace=True)
        for f in FAMS:
            draws[f][b] = _panel_slope(panels[f], samp)
    res = {}
    for f in FAMS:
        v = draws[f][~np.isnan(draws[f])]
        res[f] = dict(slope=point[f], lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5))
    # between-panel contrast: does the gradient survive issue resampling?
    contrasts = {}
    for a, b in [("explicit_political", "implicit_political"),
                 ("explicit_demographic", "implicit_political"),
                 ("implicit_political", "implicit_demographic")]:
        diff = draws[a] - draws[b]
        diff = diff[~np.isnan(diff)]
        contrasts[(a, b)] = dict(diff=point[a] - point[b],
                                 lo=np.percentile(diff, 2.5),
                                 hi=np.percentile(diff, 97.5),
                                 # share of draws with the sign reversed
                                 p_sign=float(np.mean(diff <= 0)))
    return res, contrasts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = issue_level_pairs()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in FAMS:
        g = d[d.cue_family == f]
        rows.append(dict(cue_family=f, panel=FAM_PLAIN[f], n_paired_obs=len(g),
                         n_cue_groups=g.cue_group.nunique(), **gap_test(d, f)))
    gaps = pd.DataFrame(rows)

    slopes, contrasts = slope_boot(d, args.boot, rng)
    gaps["slope"] = [slopes[f]["slope"] for f in FAMS]
    gaps["slope_lo"] = [slopes[f]["lo"] for f in FAMS]
    gaps["slope_hi"] = [slopes[f]["hi"] for f in FAMS]
    gaps.to_csv(OUT / "prediction_write_gap.csv", index=False)

    # ---------------- markdown (working notes) ----------------
    md = ["## Predicted vs. written: inference at the issue level", "",
          f"Bootstrap draws: {args.boot}, seed {args.seed}. "
          f"Issue clusters: {gaps.n_clusters.iloc[0]}.", "",
          "| Panel | obs | mean \\|predicted\\| | mean \\|written\\| | gap [95% CI] | t | p | slope β [95% CI] |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in gaps.iterrows():
        md.append(f"| {r.panel} | {r.n_paired_obs} "
                  f"| {r.abs_pred:.3f} [{r.abs_pred_lo:.3f}, {r.abs_pred_hi:.3f}] "
                  f"| {r.abs_writ:.3f} [{r.abs_writ_lo:.3f}, {r.abs_writ_hi:.3f}] "
                  f"| {r.gap:.3f} [{r.gap_lo:.3f}, {r.gap_hi:.3f}] "
                  f"| {r.t:.1f} | {r.p:.1e} | {r.slope:.2f} [{r.slope_lo:.2f}, {r.slope_hi:.2f}] |")
    md += ["", "### Between-panel slope contrasts (the transmission gradient)", "",
           "| Contrast | Δβ [95% CI] | share of draws with sign reversed |", "|---|--:|--:|"]
    for (a, b), c in contrasts.items():
        md.append(f"| {FAM_PLAIN[a]} − {FAM_PLAIN[b]} | {c['diff']:.2f} "
                  f"[{c['lo']:.2f}, {c['hi']:.2f}] | {c['p_sign']:.3f} |")
    (OUT / "prediction_write_gap.md").write_text("\n".join(md) + "\n")

    # ---------------- LaTeX (appendix) ----------------
    tex = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Predicted versus written shift, by cue type}",
        r"\label{tab:predwrite}",
        # four numeric columns each carrying a point estimate and a CI overruns the
        # 12pt text block at \small/7pt (71pt overfull); footnotesize + 4pt clears it
        # while keeping 3 d.p., which the written column needs (0.080 vs 0.061).
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        (r"\textbf{Cue type} "
         r"& \shortstack{$|\widehat{\Delta}^{\text{pred}}|$ \\ mean [95\% CI]} "
         r"& \shortstack{$|\widehat{\Delta}^{\text{writ}}|$ \\ mean [95\% CI]} "
         r"& \shortstack{\textbf{Difference} \\ mean [95\% CI]} "
         r"& \shortstack{\textbf{Slope} $\beta$ \\ $[95\%$ CI$]$} \\"),
        r"\midrule",
    ]
    for _, r in gaps.iterrows():
        tex.append(rf"{FAM_LABEL[r.cue_family]} & "
                   rf"{r.abs_pred:.3f} $[{r.abs_pred_lo:.3f},\,{r.abs_pred_hi:.3f}]$ & "
                   rf"{r.abs_writ:.3f} $[{r.abs_writ_lo:.3f},\,{r.abs_writ_hi:.3f}]$ & "
                   rf"{r.gap:.3f} $[{r.gap_lo:.3f},\,{r.gap_hi:.3f}]$ & "
                   rf"{r.slope:.2f} $[{r.slope_lo:.2f},\,{r.slope_hi:.2f}]$ \\")
    tex += [
        r"\bottomrule", r"\end{tabular}", "",
        r"\vspace{3pt}",
        r"\begin{minipage}{0.94\linewidth}",
        (r"\footnotesize\textit{Note:} Both quantities are differenced against the same "
         r"issue's no-cue baseline. The middle column is a paired comparison of absolute "
         rf"shift magnitudes, averaged within issue and then over the {gaps.n_clusters.iloc[0]} "
         r"issues, with a $t$-interval on the issue clusters ($\mathrm{df}=18$); it tests "
         r"whether a cue is written more weakly than it is predicted. The right-hand column "
         r"is the descriptive slope of written on predicted shift plotted in "
         r"Figure~\ref{fig:belief-stance}, with a bootstrap-over-issues confidence interval "
         rf"({args.boot} draws). Two limits apply. The predicted shift is a continuous "
         r"$0$--$100$ rescaling while the written shift derives from three-category stance "
         r"labels, so part of every gap is a scale artifact rather than hedging (see the "
         r"discretisation check, Appendix~\ref{robustness}); and because absolute values are "
         r"taken before averaging, a panel whose true shifts are near zero has its gap "
         r"inflated by noise, so the name row should be read as an upper bound. The slope "
         r"is fitted on cue group $\times$ model means, which are not mutually independent, "
         r"so its interval reflects issue sampling only."),
        r"\end{minipage}", r"\end{table}",
    ]
    (OUT / "prediction_write_gap.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\nWrote {OUT}/prediction_write_gap.{{csv,md,tex}}")


if __name__ == "__main__":
    main()
