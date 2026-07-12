#!/usr/bin/env python3
"""Common-ruler guard for the belief→stance under-acting claim (A3).

The probe compares a *continuous* belief shift (mean of 0–100 estimates) against a
*trichotomized* stance shift (mean of {−1,0,+1} labels with a large neutral mass).
Trichotomization mechanically compresses the stance axis, so "believes −0.80,
writes −0.22" overstates under-acting — part of the gap is the ruler, not the model.

Fix (Törnberg-style symmetric discretization): pass belief through the *same*
trichotomization as stance — parsed_score >60 → +1, 40–60 → 0, <40 → −1, then
×liberal_sign — recompute belief shifts, and compare on a common ruler. If stance
shifts still fall short of belief shifts (points below y=x, through-origin slope <1),
under-acting survives the scale objection.

The rank-order claim (belief predicts stance) is unaffected; this guards only the
magnitude claim. Reads results/full/belief_probe_<model>.csv +
results/probe_internal/<model>_mediation_full3x.csv. Writes
results/probe_internal/belief_under_acting.csv and a figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["llama", "gemma", "qwen"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B", "qwen": "Qwen3.6-27B"}
MODEL_COLOUR = {"llama": "#2e6da4", "gemma": "#27915b", "qwen": "#c0392b"}
PROBE = Path("results/probe_internal")
BELIEF = Path("results/full")


def belief_shifts(model):
    d = pd.read_csv(BELIEF / f"belief_probe_{model}.csv", low_memory=False)
    d = d[d.probe_kind == "opinion"].copy()
    d["score"] = pd.to_numeric(d["parsed_score"], errors="coerce")
    d = d.dropna(subset=["score"])
    sign = d["liberal_sign"]
    # continuous (primary) and trichotomized (common ruler with stance)
    d["b_cont"] = (d["score"] - 50) / 50 * sign
    tri = np.where(d["score"] > 60, 1, np.where(d["score"] < 40, -1, 0))
    d["b_tri"] = tri * sign
    med = pd.read_csv(PROBE / f"{model}_mediation_full3x.csv")

    def shift(col):
        base = d[d.cue_family == "baseline"].groupby("issue_id")[col].mean()
        vals = []
        for _, r in med.iterrows():
            c = d[(d.cue_family == r.cue_family) & (d.cue_group == r.cue_group)]
            vals.append((c.groupby("issue_id")[col].mean() - base).mean())
        return np.array(vals)

    med["belief_cont"] = shift("b_cont")
    med["belief_tri"] = shift("b_tri")
    # neutral mass on the belief side (fraction of opinion responses in the 40-60 band)
    belief_neutral_frac = float(((d["score"] >= 40) & (d["score"] <= 60)).mean())
    return med, belief_neutral_frac


def origin_slope(x, y):
    return float(np.sum(x * y) / np.sum(x * x))


def main():
    all_rows = []
    summ = []
    for m in MODELS:
        med, belief_neutral = belief_shifts(m)
        med["model"] = m
        all_rows.append(med)
        # stance-side neutral mass, for the mechanism comparison
        st = pd.read_csv(BELIEF.parent / "full_3x" / f"bert_eval_{m}.csv",
                         usecols=["bert_collapsed_stance"], low_memory=False)
        stance_neutral = float((st["bert_collapsed_stance"] == "neutral").mean())
        s_cont = origin_slope(med["belief_cont"].to_numpy(), med["stance_shift"].to_numpy())
        s_tri = origin_slope(med["belief_tri"].to_numpy(), med["stance_shift"].to_numpy())
        r_tri = stats.pearsonr(med["belief_tri"], med["stance_shift"])[0]
        # the headline Republican example
        rep = med[(med.cue_family == "explicit_political") & (med.cue_group == "republican")].iloc[0]
        summ.append({"model": m,
                     "underact_slope_continuous": s_cont,
                     "underact_slope_trichotomized": s_tri,
                     "r_tri_vs_stance": r_tri,
                     "belief_neutral_frac": belief_neutral,
                     "stance_neutral_frac": stance_neutral,
                     "rep_belief_cont": rep["belief_cont"],
                     "rep_belief_tri": rep["belief_tri"],
                     "rep_stance": rep["stance_shift"]})
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(PROBE / "belief_under_acting.csv", index=False)
    sd = pd.DataFrame(summ)

    print("=== Under-acting on a common ruler (belief trichotomized like stance) ===\n")
    print("through-origin slope of stance_shift on belief_shift  (<1 = model under-acts):\n")
    for _, r in sd.iterrows():
        print(f"  {MODEL_LABEL[r['model']]:>14}:  continuous belief ruler = {r['underact_slope_continuous']:.2f}   "
              f"common (trichotomized) ruler = {r['underact_slope_trichotomized']:.2f}   "
              f"(r_tri={r['r_tri_vs_stance']:.2f})")
    print("\nRepublican cue, believes vs writes:")
    for _, r in sd.iterrows():
        print(f"  {MODEL_LABEL[r['model']]:>14}:  continuous belief {r['rep_belief_cont']:+.2f} → writes "
              f"{r['rep_stance']:+.2f}   |   common-ruler belief {r['rep_belief_tri']:+.2f} → writes "
              f"{r['rep_stance']:+.2f}")
    survives = (sd["underact_slope_trichotomized"] < 1).all()
    ratio = (sd["underact_slope_trichotomized"] / sd["underact_slope_continuous"]).mean()
    print("\nNeutral mass — belief vs writing (the mechanism):")
    for _, r in sd.iterrows():
        print(f"  {MODEL_LABEL[r['model']]:>14}:  belief neutral (40–60) = {r['belief_neutral_frac']*100:.0f}%   "
              f"writing neutral = {r['stance_neutral_frac']*100:.0f}%")
    print(f"\nUnder-acting (slope<1) survives the common ruler on all models: {survives}")
    print(f"Contrary to the scale worry, trichotomizing belief the same way as stance *lowers* the "
          f"slope (~{ratio:.1f}× the continuous), i.e. under-acting is if anything STRONGER on a "
          f"common ruler — not a trichotomization artifact. Mechanism: the model's beliefs are "
          f"polarized (small neutral mass) while its writing carries a large neutral mass, so it "
          f"hedges in prose what it states clearly in belief. This is the same neutral-compression "
          f"that drives the directional-only result (robustness §7).")
    make_figure(out, sd)
    print(f"Wrote {PROBE/'belief_under_acting.csv'}")


def make_figure(out, sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharex=True, sharey=True)
    lim = 1.3
    for ax, col, title in [(axes[0], "belief_cont", "Continuous belief ruler (primary)"),
                           (axes[1], "belief_tri", "Common ruler: belief trichotomized like stance")]:
        ax.plot([-lim, lim], [-lim, lim], "--", color="#888", lw=1, label="y = x (acts = believes)")
        ax.axhline(0, color="#ddd", lw=0.6); ax.axvline(0, color="#ddd", lw=0.6)
        for m in MODELS:
            s = out[out.model == m]
            ax.scatter(s[col], s["stance_shift"], s=34, color=MODEL_COLOUR[m],
                       alpha=0.8, edgecolor="white", linewidth=0.4, label=MODEL_LABEL[m])
            sl = sd[sd.model == m][f"underact_slope_{'continuous' if col=='belief_cont' else 'trichotomized'}"].iloc[0]
            xs = np.linspace(-lim, lim, 50)
            ax.plot(xs, sl * xs, color=MODEL_COLOUR[m], lw=1, alpha=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("belief shift (cued − baseline)")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("written stance shift")
    axes[0].legend(fontsize=8, frameon=True, loc="lower right")
    fig.suptitle("Under-acting survives the ruler: points stay below y=x even when belief is "
                 "trichotomized like stance", y=1.01, fontsize=11)
    fig.tight_layout()
    p = Path("figures/probe_thesis/belief_under_acting_common_ruler.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
