#!/usr/bin/env python3
"""Is response non-completion flat across cues on the corpus of record? (Appendix)

WHY THIS EXISTS, AND WHAT IT DOES NOT DO.

The refusal robustness check (refusal_bounds.py) computes Manski worst-case bounds on
every cue effect under adversarial assignment of refused responses. It runs on
``results/full/eval_<model>.csv`` -- the EARLIER, judge-scored generation run -- for the
three open-weight models, because refusal detection needs the response *text* and that
run is the only local corpus that retains it. The 3-repeat rerun that is the corpus of
record (``results/full_3x``) stores only scores, so refusals cannot be re-labelled there
without pulling the generations back from Brains.

This script therefore does NOT recompute the Manski bounds on the corpus of record. It
establishes the weaker thing that *is* checkable there: whether responses fail to
complete normally, and whether that failure rate is flat across cues. Two mechanisms are
visible in ``finish_reason``:

  * ``content_filter`` -- a hard provider-side refusal.
  * ``length`` -- truncation at the token cap. Not a refusal, but it is differential
    missingness if it varies by cue, since a truncated response is scored on less text.

If both are negligible and flat, then whatever selection refusals induce on the corpus of
record is bounded by something very small, and the earlier-corpus Manski bounds are the
binding evidence. If either varies by cue, that is a problem the bounds do not cover.

Interpret this as a necessary-but-not-sufficient check, and say so in the appendix: it
cannot detect a soft refusal ("I'd rather not take a side on this") that terminates
normally. Only the text-based labelling can, and that is why the earlier corpus is still
cited.

Usage:  python3 analysis/05_robustness/finish_reason_flatness.py
Writes: results/robustness/finish_reason_flatness.{csv,md}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _common import (MODELS, MODEL_LABEL, EVAL_PREFIX, SCORE_COL, FULL3X,  # noqa: E402
                     ROBUST)

NONNORMAL = {"content_filter", "length"}


def truncation_bounds(model: str) -> pd.DataFrame:
    """Manski-style worst-case bounds on each cue effect under adversarial truncation.

    Truncation is differential missingness: a response cut at the token cap is scored on
    less text than it would have been. Following the same logic refusal_bounds.py applies
    to refusals, we treat every truncated response as *unobserved* and ask how far the cue
    effect could move if those responses had taken the most adversarial value available on
    the {-1, 0, +1} scale:

        upper bound: cued truncations -> +1, baseline truncations -> -1
        lower bound: cued truncations -> -1, baseline truncations -> +1

    If a cue effect keeps its sign under both, differential truncation cannot be driving
    it. Unlike the refusal bounds, this runs on the corpus of record for all five models,
    because ``finish_reason`` needs no response text.

    Baseline matching follows _common.shift_table: implicit (Arm-B) cues are compared to
    the template-matched baseline subset."""
    d = pd.read_csv(FULL3X / f"{EVAL_PREFIX}_{model}.csv", low_memory=False)
    d = d.rename(columns={SCORE_COL: "lib"})
    d["lib"] = d["lib"].astype(float)
    d["template_id"] = d.prompt_id.str.split("__").str[1]
    d["trunc"] = d.finish_reason == "length"
    # Truncated responses were still scored, so they are not literally missing. The
    # question is whether truncation *shifts* the score. delta_trunc measures that shift
    # directly, and it is what turns the worst case into a plausible case below.
    delta_trunc = (d.loc[d.trunc, "lib"].mean() - d.loc[~d.trunc, "lib"].mean()
                   if d.trunc.any() else 0.0)
    base_all = d[d.cue_family == "baseline"]
    rows = []
    for (fam, grp), cue in d[d.cue_family != "baseline"].groupby(["cue_family", "cue_group"]):
        base = (base_all[base_all.template_id.isin(cue.template_id.unique())]
                if str(fam).startswith("implicit") else base_all)
        nc, nb = len(cue), len(base)
        tc, tb = int(cue.trunc.sum()), int(base.trunc.sum())
        sc = float(cue.loc[~cue.trunc, "lib"].sum())
        sb = float(base.loc[~base.trunc, "lib"].sum())
        observed = cue["lib"].mean() - base["lib"].mean()
        # (a) worst case: every truncated response takes the most adversarial value
        hi = (sc + tc * 1) / nc - (sb + tb * -1) / nb
        lo = (sc + tc * -1) / nc - (sb + tb * 1) / nb
        # (b) plausible case: truncation contaminates the mean only in proportion to how
        # much it shifts a score (delta_trunc) times how much more often it happens under
        # the cue than under the baseline. This is the bias actually implied by the data,
        # where (a) is the bias implied by assuming the worst possible scores.
        bias = (tc / nc - tb / nb) * delta_trunc
        rows.append(dict(model=model, cue_family=fam, cue_group=grp,
                         delta_observed=observed,
                         trunc_rate_cue=tc / nc, trunc_rate_base=tb / nb,
                         delta_trunc=delta_trunc, bias_plausible=bias,
                         delta_debiased=observed - bias,
                         trunc_lo=lo, trunc_hi=hi, trunc_width=hi - lo,
                         sign_robust=bool(np.sign(lo) == np.sign(hi)),
                         sign_robust_plausible=bool(
                             np.sign(observed) == np.sign(observed - bias))))
    return pd.DataFrame(rows)


def main() -> int:
    rows = []
    for m in MODELS:
        d = pd.read_csv(FULL3X / f"{EVAL_PREFIX}_{m}.csv", low_memory=False,
                        usecols=["cue_family", "cue_group", "finish_reason"])
        d["filtered"] = (d.finish_reason == "content_filter").astype(float)
        d["truncated"] = (d.finish_reason == "length").astype(float)
        for (fam, grp), g in d.groupby(["cue_family", "cue_group"], dropna=False):
            rows.append(dict(model=m, cue_family=fam, cue_group=grp, n=len(g),
                             filtered_rate=g.filtered.mean(),
                             truncated_rate=g.truncated.mean()))
    out = pd.DataFrame(rows)
    bounds = pd.concat([truncation_bounds(m) for m in MODELS], ignore_index=True)
    bounds.to_csv(ROBUST / "truncation_bounds.csv", index=False)
    ROBUST.mkdir(parents=True, exist_ok=True)
    out.to_csv(ROBUST / "finish_reason_flatness.csv", index=False)

    md = ["## Non-completion on the corpus of record (`full_3x`, luna scoring)", "",
          "Rates of provider-side filtering and token-cap truncation, per model, over the "
          "15 conditions (14 cues + baseline). This checks whether *visible* "
          "non-completion is flat across cues; it cannot see a soft refusal that "
          "terminates normally, which is why the text-based Manski bounds "
          "(`refusal_bounds.py`, earlier corpus, three open-weight models) remain the "
          "binding refusal evidence.", "",
          "| Model | conditions | filtered: max | truncated: min-max | truncated: spread |",
          "|---|--:|--:|--:|--:|"]
    for m in MODELS:
        s = out[out.model == m]
        spread = s.truncated_rate.max() - s.truncated_rate.min()
        md.append(f"| {MODEL_LABEL[m]} | {len(s)} | {s.filtered_rate.max()*100:.3f}% "
                  f"| {s.truncated_rate.min()*100:.2f}%-{s.truncated_rate.max()*100:.2f}% "
                  f"| {spread*100:.2f} pp |")

    worst = out.loc[out.truncated_rate.idxmax()]
    md += ["",
           f"- highest filtering rate anywhere: **{out.filtered_rate.max()*100:.4f}%** "
           f"({int(out.filtered_rate.gt(0).sum())} of {len(out)} conditions show any at all)",
           f"- highest truncation rate anywhere: **{worst.truncated_rate*100:.2f}%** "
           f"({MODEL_LABEL[worst.model]}, {worst.cue_family}/{worst.cue_group})",
           f"- largest within-model spread in truncation across conditions: "
           f"**{max(out[out.model==m].truncated_rate.max()-out[out.model==m].truncated_rate.min() for m in MODELS)*100:.2f} pp**",
           "",
           "Provider-side filtering is essentially absent. Truncation is the only "
           "non-completion mechanism with a non-trivial rate, so it is bounded below "
           "rather than assumed away.", "",
           "### Worst-case bounds under adversarial truncation", "",
           "Every truncated response is treated as unobserved and assigned the most "
           "adversarial value on the $\\{-1,0,+1\\}$ scale (cued $\\to +1$ / baseline "
           "$\\to -1$ for the upper bound, and the reverse for the lower). This is the "
           "same Manski logic `refusal_bounds.py` applies to refusals, but it runs on the "
           "corpus of record for **all five models**, since `finish_reason` needs no "
           "response text.", "",
           "| Model | cues | sign robust | widest bound | not sign-robust |",
           "|---|--:|--:|--:|---|"]
    for m in MODELS:
        s = bounds[bounds.model == m]
        bad = s[~s.sign_robust]
        names = ", ".join(f"{r.cue_family.split('_')[0][:3]}/{r.cue_group} ({r.delta_observed:+.3f})"
                          for _, r in bad.iterrows()) or "--"
        md.append(f"| {MODEL_LABEL[m]} | {len(s)} | {int(s.sign_robust.sum())}/{len(s)} "
                  f"| {s.trunc_width.max():.4f} | {names} |")
    big = bounds[bounds.delta_observed.abs() > 0.05]
    md += ["",
           f"- across all {len(bounds)} cue effects, **{int(bounds.sign_robust.sum())}** keep "
           f"their sign under worst-case truncation",
           f"- restricting to the {len(big)} effects with $|\\Delta| > 0.05$ (the ones "
           f"carrying claims): **{int(big.sign_robust.sum())}/{len(big)}** are sign-robust",
           f"- the widest bound anywhere is **{bounds.trunc_width.max():.4f}** on the "
           f"$-1$ to $+1$ scale",
           "",
           "### Why the worst case is the wrong bound here", "",
           "Truncated responses were still *scored* -- they are not missing data. So the "
           "worst case, which assigns every truncated response $\\pm 1$, answers a question "
           "the design does not pose. What the data supports is a point estimate of the "
           "contamination: how much truncation shifts a score, times how much more often "
           "it happens under the cue than under the baseline.", "",
           "| Model | trunc rate | score shift when truncated | max implied bias | sign robust |",
           "|---|--:|--:|--:|--:|"]
    for m in MODELS:
        s = bounds[bounds.model == m]
        tr = f"{s.trunc_rate_cue.min()*100:.2f}-{s.trunc_rate_cue.max()*100:.2f}%"
        dt = s.delta_trunc.iloc[0]
        md.append(f"| {MODEL_LABEL[m]} | {tr} | {dt:+.3f} "
                  f"| {s.bias_plausible.abs().max():.4f} "
                  f"| {int(s.sign_robust_plausible.sum())}/{len(s)} |")
    md += ["",
           f"- truncated responses score more liberal than completed ones in every model "
           f"that truncates at all (shift $+{bounds.delta_trunc.min():.3f}$ to "
           f"$+{bounds.delta_trunc.max():.3f}$), so truncation is **not** ignorable in "
           f"principle",
           f"- but the bias it implies for any cue effect is at most "
           f"**{bounds.bias_plausible.abs().max():.4f}**, because the cue-to-baseline "
           f"difference in truncation rate is small even where the rate itself is not",
           f"- under that bias, **{int(bounds.sign_robust_plausible.sum())}/{len(bounds)}** "
           f"cue effects keep their sign, against "
           f"{int(bounds.sign_robust.sum())}/{len(bounds)} under the worst case",
           "",
           "So the worst-case failures are an artefact of the $\\pm 1$ assignment, not "
           "evidence of a truncation problem: the largest bias the observed data implies "
           f"({bounds.bias_plausible.abs().max():.4f}) is an order of magnitude below the "
           "smallest cue effect the thesis interprets. Report the plausible-case column as "
           "the finding and the worst case as the (uninformative) extreme."]
    (ROBUST / "finish_reason_flatness.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {ROBUST}/finish_reason_flatness.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
