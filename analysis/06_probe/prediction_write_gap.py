#!/usr/bin/env python3
"""Inference for the predicted-vs-written figure (§4.4, fig:belief-stance).

The figure is descriptive: one point per cue group x model, so 15-20 points per
panel. That aggregation is a plotting choice, not the design -- the *pairing* lives at
the issue level, because every cue x issue cell has both an elicited opinion prediction
and a written stance for the same issue. This script does the inference at that level
and emits the appendix table.

Two distinct claims, deliberately kept apart because the data supports them unequally:

  (1) UNDER-WRITING, within a cue type. The transmission slope beta of written shift on
      predicted shift, tested against **beta = 1** -- one unit of predicted shift
      becoming one unit of written shift. Estimated at the issue level with a fixed
      effect per model, so beta is a within-model rate. Holds for all four cue types.

  (2) THE TRANSMISSION GRADIENT, between cue types. Differences in beta across cue
      types, with a clustered bootstrap over issues. Survives for party label and
      race x gender against state; does NOT survive for state against name.

Testing against 1 rather than differencing |predicted| and |written| is deliberate:
beta is signed throughout, so it avoids the bias that absolute values introduce for
quantities whose true value is near zero, and beta = 1 is the same reference the RQ2
calibration slope uses. The absolute magnitude columns are retained as description --
they are what separates the state cue from the name cue, since the slopes do not.

Caveats that the numbers do not remove, and that the appendix note repeats:

  * x is a continuous 0-100 rescaling, y is a mean over three-category stance labels,
    so part of any shortfall below 1 is a scale artifact rather than hedging. The
    discretisation check (Appendix, robustness) is what speaks to that.
  * The two magnitude columns take |.| before averaging, which inflates a quantity
    whose true value is near zero. Read the name row's magnitudes as upper bounds; the
    beta column is unaffected.
  * Uncertainty is issue sampling only. Model is a fixed effect, so nothing here
    generalises beyond these five systems.

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
    """Transmission slope for one cue type, with MODEL ABSORBED AS A FIXED EFFECT.

    Regress the written shift on the predicted shift at the issue level, including a
    dummy per model (reference model dropped). The five models are chosen systems, not
    a sample from a population of models, so they enter as fixed effects -- Week 8's
    random-effects assumption alpha_i ~ N(0, sigma^2) requires the levels to be draws,
    which these are not. An earlier version pooled the models into one slope, which
    silently treated them as exchangeable units.

    Absorbing model also removes between-model differences in overall responsiveness
    from the slope, so beta is a *within-model* transmission rate: of a unit of
    predicted shift, how much does that model write?"""
    if issues is not None:
        # index-based take so an issue drawn twice contributes twice (clustered bootstrap)
        g = pd.concat([g[g.issue_id == i] for i in issues], ignore_index=True)
    if len(g) < 8 or g.p_shift.std() == 0:
        return np.nan
    # design matrix: intercept, predicted shift, model dummies (first model dropped)
    models = sorted(g.model.unique())
    X = [np.ones(len(g)), g.p_shift.to_numpy()]
    for m in models[1:]:
        X.append((g.model == m).to_numpy(float))
    X = np.column_stack(X)
    try:
        beta, *_ = np.linalg.lstsq(X, g.w_shift.to_numpy(), rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[1])          # coefficient on the predicted shift


def slope_boot(d: pd.DataFrame, n_boot: int, rng: np.random.Generator):
    """Transmission slope per cue type, tested against 1, plus between-type contrasts.

    beta = 1 is the reference of interest, not beta = 0: "the model writes what it
    predicts" means a unit of predicted shift becomes a unit of written shift. Testing
    against 1 states under-writing directly and avoids the earlier |predicted| minus
    |written| formulation, whose absolute values inflate any quantity whose true value
    is near zero. It is also the same reference the RQ2 calibration slope uses.

    Uncertainty is a clustered bootstrap over the 19 issues (Week 2's bootstrap CI with
    the issue as the resampling unit, matching every other interval in the thesis)."""
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
        lo, hi = np.percentile(v, [2.5, 97.5])
        # bootstrap two-sided p for H0: beta = 1, as the share of draws at least as far
        # from 1 as the point estimate is, doubled (Week 3's bootstrap test logic)
        p1 = 2 * min(np.mean(v >= 1.0), np.mean(v <= 1.0))
        res[f] = dict(slope=point[f], lo=lo, hi=hi,
                      excludes_one=bool(hi < 1.0 or lo > 1.0),
                      p_vs_one=float(min(p1, 1.0)))
    contrasts = {}
    for a, b in [("explicit_political", "implicit_political"),
                 ("explicit_demographic", "implicit_political"),
                 ("implicit_political", "implicit_demographic")]:
        diff = draws[a] - draws[b]
        diff = diff[~np.isnan(diff)]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p0 = 2 * min(np.mean(diff >= 0), np.mean(diff <= 0))
        contrasts[(a, b)] = dict(diff=point[a] - point[b], lo=lo, hi=hi,
                                 p_vs_zero=float(min(p0, 1.0)))
    return res, contrasts


def fmt_p(p: float, n_boot: int, tex: bool = False) -> str:
    """Format a bootstrap p at the resolution the bootstrap actually has.

    A two-sided bootstrap p over B draws cannot be smaller than 2/B, so printing
    "0.0000" from 5,000 draws claims precision the method does not have. Report
    "< 2/B" instead."""
    floor = 2.0 / n_boot
    if p < floor:
        return rf"$<${floor:.4f}" if tex else f"<{floor:.4f}"
    return f"{p:.4f}"


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (q-values), monotone.

    BH rather than the Holm-Bonferroni of Week 4 for the same reason as the RQ1 family:
    with several genuinely small true effects, family-wise control spends too much power.
    Applied here so this table's family is corrected on the same basis as RQ1's rather
    than being the one uncorrected set of tests in the thesis."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):          # step up from the largest p
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = running
    return list(q)


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
    for k in ("slope", "lo", "hi", "p_vs_one", "excludes_one"):
        gaps["slope" if k == "slope" else f"slope_{k}"] = [slopes[f][k] for f in FAMS]

    # One BH family for this table: the 4 slope-vs-1 tests plus the 3 between-type
    # contrasts. Corrected together so this table is not the single uncorrected set of
    # tests in the thesis; RQ1's 70-cell family is corrected separately.
    fam_p = list(gaps["slope_p_vs_one"]) + [c["p_vs_zero"] for c in contrasts.values()]
    fam_q = bh_adjust(fam_p)
    gaps["slope_q_vs_one"] = fam_q[:len(FAMS)]
    contrast_q = dict(zip(contrasts.keys(), fam_q[len(FAMS):]))
    gaps.to_csv(OUT / "prediction_write_gap.csv", index=False)

    # ---------------- markdown (working notes) ----------------
    md = ["## Predicted vs. written: inference at the issue level", "",
          f"Clustered bootstrap over issues: {args.boot} draws, seed {args.seed}, "
          f"{gaps.n_clusters.iloc[0]} issue clusters. Model enters each slope as a fixed "
          "effect, so beta is a within-model transmission rate.", "",
          "`beta = 1` is the reference: a unit of predicted shift becoming a unit of "
          "written shift. BH q-values are over this table's family of 7 tests.", "",
          "| Cue type | obs | mean \\|predicted\\| | mean \\|written\\| | β [95% CI] | β≠1? | p | BH q |",
          "|---|--:|--:|--:|--:|:--:|--:|--:|"]
    for _, r in gaps.iterrows():
        md.append(f"| {r.panel} | {r.n_paired_obs} "
                  f"| {r.abs_pred:.3f} [{r.abs_pred_lo:.3f}, {r.abs_pred_hi:.3f}] "
                  f"| {r.abs_writ:.3f} [{r.abs_writ_lo:.3f}, {r.abs_writ_hi:.3f}] "
                  f"| {r.slope:.2f} [{r.slope_lo:.2f}, {r.slope_hi:.2f}] "
                  f"| {'yes' if r.slope_excludes_one else 'no'} "
                  f"| {fmt_p(r.slope_p_vs_one, args.boot)} "
                  f"| {fmt_p(r.slope_q_vs_one, args.boot)} |")
    md += ["", "### Between-cue-type slope contrasts (the transmission gradient)", "",
           "| Contrast | Δβ [95% CI] | p | BH q |", "|---|--:|--:|--:|"]
    for (a, b), c in contrasts.items():
        md.append(f"| {FAM_PLAIN[a]} − {FAM_PLAIN[b]} | {c['diff']:.2f} "
                  f"[{c['lo']:.2f}, {c['hi']:.2f}] "
                  f"| {fmt_p(c['p_vs_zero'], args.boot)} "
                  f"| {fmt_p(contrast_q[(a, b)], args.boot)} |")
    (OUT / "prediction_write_gap.md").write_text("\n".join(md) + "\n")

    # ---------------- LaTeX (appendix) ----------------
    tex = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Predicted versus written shift, by cue type}",
        r"\label{tab:predwrite}",
        # four numeric columns each carrying a point estimate and a CI overruns the
        # 12pt text block at \small/7pt (71pt overfull); footnotesize + 4pt clears it
        # while keeping 3 d.p., which the written column needs (0.080 vs 0.061).
        r"\small",
        r"\setlength{\tabcolsep}{9pt}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        (r"\textbf{Cue type} "
         r"& $|\widehat{\Delta}^{\text{pred}}|$ "
         r"& $|\widehat{\Delta}^{\text{writ}}|$ "
         r"& \textbf{Transmission} $\beta$ "
         r"& BH $q$ ($\beta \neq 1$) \\"),
        r"\midrule",
    ]
    # Point estimate on the cue's row, 95% interval on the row beneath it in brackets --
    # the regression-table convention (cf. the AAS summative's estimate-over-SE layout).
    # Putting both in one cell made five columns of "0.475 [0.441, 0.510]", which reads
    # as a wall of digits.
    for i, (_, r) in enumerate(gaps.iterrows()):
        if i:
            tex.append(r"\addlinespace[2pt]")
        tex.append(rf"{FAM_LABEL[r.cue_family]} & {r.abs_pred:.3f} & {r.abs_writ:.3f} & "
                   rf"{r.slope:.2f} & {fmt_p(r.slope_q_vs_one, args.boot, tex=True)} \\")
        tex.append(rf" & \scriptsize $[{r.abs_pred_lo:.3f},\,{r.abs_pred_hi:.3f}]$ "
                   rf"& \scriptsize $[{r.abs_writ_lo:.3f},\,{r.abs_writ_hi:.3f}]$ "
                   rf"& \scriptsize $[{r.slope_lo:.2f},\,{r.slope_hi:.2f}]$ & \\")
    tex += [
        r"\bottomrule", r"\end{tabular}", "",
        r"\vspace{3pt}",
        r"\begin{minipage}{0.94\linewidth}",
        (r"\footnotesize\textit{Note:} Both quantities are differenced against the same "
         r"issue's no-cue baseline, so issue difficulty cancels. The two magnitude columns "
         rf"are means of absolute shifts, averaged within issue and then over the "
         rf"{gaps.n_clusters.iloc[0]} issues, with $t$-intervals on the issue clusters "
         r"($\mathrm{df}=18$). $\beta$ is the transmission slope of written on predicted "
         r"shift, estimated at the issue level with a fixed effect per model, so it is a "
         r"within-model rate; the reference of interest is $\beta = 1$, one unit of "
         r"predicted shift becoming one unit of written shift, and $q$ is the "
         r"Benjamini--Hochberg adjusted $p$ for $\beta \neq 1$ over this table's family of "
         rf"seven tests. Intervals are clustered bootstraps over issues ({args.boot} draws). "
         r"Two limits remain. The predicted shift is a continuous $0$--$100$ rescaling while "
         r"the written shift derives from three-category stance labels, so part of any "
         r"shortfall is a scale artifact rather than hedging (see the discretisation check, "
         r"Appendix~\ref{robustness}); and because the magnitude columns take absolute "
         r"values before averaging, a cue whose true shifts are near zero has those columns "
         r"inflated by noise, so the name row should be read as an upper bound. The "
         r"$\beta$ column is unaffected by that second issue, as it is signed throughout."),
        r"\end{minipage}", r"\end{table}",
    ]
    (OUT / "prediction_write_gap.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\nWrote {OUT}/prediction_write_gap.{{csv,md,tex}}")


if __name__ == "__main__":
    main()
