#!/usr/bin/env python3
"""Composition-level (variance-collapse) test of the flattening hypothesis.

Wang et al.'s flattening failure includes *variance collapse* — a model treating a
heterogeneous group as a point. A calibration analysis on means alone cannot see
this: a model can match the group mean while writing one side almost every time.

We cannot map model responses to individuals, but we can compare **compositions**.
For each cue group and issue:

  - CES gives the real weighted proportion of that subgroup on the liberal side
    (forced choice, so among the opinionated by construction).
  - The model gives the proportion of its *directional* (non-neutral) responses
    written on the liberal side.

Restricting to opinionated responses on both sides sidesteps the neutral-mapping
problem entirely (it is a proportion among those who take a side). A model that
writes 95% liberal for Democrat-cued users when 80% of real Democrats hold the
liberal position has collapsed within-group diversity even though the mean looks
calibrated.

Diagnostics per (model, cue group):
  * calibration of composition: OLS slope of model share on CES share across issues
  * **collapse index** = mean extremity gap = mean_issues(|p_model-0.5| - |p_ces-0.5|)
    > 0  => model is more one-sided than the real group (variance collapse)
  * share of issues where the model is strictly more extreme than the population

Reads the CES microdata + results/full_3x/bert_eval_*.csv. Writes a per-issue
table, a per-group summary, and a figure.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, MODEL_COLOUR, ROBUST, load_all

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"
ISSUES = "data/input/issues_experiment.csv"
STATES = "data/input/states/state_bank.csv"


def leading_ints(cell: str) -> list[int]:
    out = []
    for seg in str(cell).split(";"):
        m = re.match(r"\s*(\d+)", seg)
        if m:
            out.append(int(m.group(1)))
    return out


def ces_liberal_shares() -> pd.DataFrame:
    """Weighted liberal-side share per (cue_family, cue_group, ces_variable)."""
    issues = pd.read_csv(ISSUES)
    issues = issues[issues["analysis_tier"] == "main"].copy()
    issue_vars = issues["ces_variable"].tolist()
    raw = pd.read_stata(CES_DTA, columns=["commonweight", "pid3", "race", "gender4"] + issue_vars,
                        convert_categoricals=False)
    st = pd.read_stata(CES_DTA, columns=["inputstate"], convert_categoricals=True)
    raw["state"] = st["inputstate"].astype(str).values
    w = raw["commonweight"].astype(float)

    lib = pd.DataFrame(index=raw.index)
    for _, row in issues.iterrows():
        v = row["ces_variable"]
        sign = int(row["liberal_sign"])
        sup = set(leading_ints(row["ces_support_code"]))
        opp = set(leading_ints(row["ces_oppose_code"]))
        val = pd.Series(np.nan, index=raw.index)
        val[raw[v].isin(sup)] = 1.0 * sign
        val[raw[v].isin(opp)] = -1.0 * sign
        lib[v] = val

    sb = pd.read_csv(STATES)
    state_cat = dict(zip(sb["state"], sb["category"]))
    state_cat.setdefault("District of Columbia", "blue_state")
    cat = raw["state"].map(state_cat)

    masks = {
        ("explicit_political", "democrat"): raw["pid3"] == 1,
        ("explicit_political", "republican"): raw["pid3"] == 2,
        ("explicit_political", "independent"): raw["pid3"] == 3,
        ("explicit_demographic", "white_man"): (raw["race"] == 1) & (raw["gender4"] == 1),
        ("explicit_demographic", "white_woman"): (raw["race"] == 1) & (raw["gender4"] == 2),
        ("explicit_demographic", "black_man"): (raw["race"] == 2) & (raw["gender4"] == 1),
        ("explicit_demographic", "black_woman"): (raw["race"] == 2) & (raw["gender4"] == 2),
        ("implicit_political", "blue_state"): cat == "blue_state",
        ("implicit_political", "red_state"): cat == "red_state",
        ("implicit_political", "swing_state"): cat == "swing_state",
        ("baseline", "baseline"): pd.Series(True, index=raw.index),
    }

    def lib_share(vals, weights, mask):
        v = vals.where(mask)
        opinion = v.notna()  # forced choice: all non-missing are opinionated
        denom = weights[opinion].sum()
        if denom <= 0:
            return np.nan, 0
        share = weights[opinion & (v > 0)].sum() / denom
        return float(share), int(opinion.sum())

    recs = []
    for (fam, grp), mask in masks.items():
        for v in issue_vars:
            share, n = lib_share(lib[v], w, mask)
            recs.append({"cue_family": fam, "cue_group": grp, "ces_variable": v,
                         "ces_lib_share": share, "ces_n": n})
        # name cue mirrors the same real subgroup as the demographic label
        if fam == "explicit_demographic":
            for v in issue_vars:
                share, n = lib_share(lib[v], w, mask)
                recs.append({"cue_family": "implicit_demographic", "cue_group": grp,
                             "ces_variable": v, "ces_lib_share": share, "ces_n": n})
    return pd.DataFrame(recs)


def model_liberal_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Directional (non-neutral) liberal share per (model, cue, issue)."""
    d = df.copy()
    d["opinion"] = d["y"] != 0
    d["lib"] = d["y"] > 0
    grp = d.groupby(["model", "cue_family", "cue_group", "ces_variable"])
    out = grp.agg(n_total=("y", "size"),
                  n_opinion=("opinion", "sum"),
                  n_lib=("lib", "sum")).reset_index()
    out["model_lib_share"] = out["n_lib"] / out["n_opinion"].replace(0, np.nan)
    out["neutral_rate"] = 1 - out["n_opinion"] / out["n_total"]
    return out


def main():
    df = load_all()
    mshare = model_liberal_shares(df)
    cshare = ces_liberal_shares()
    merged = mshare.merge(cshare, on=["cue_family", "cue_group", "ces_variable"], how="inner")
    merged = merged.dropna(subset=["model_lib_share", "ces_lib_share"])
    merged["model_extremity"] = (merged["model_lib_share"] - 0.5).abs()
    merged["ces_extremity"] = (merged["ces_lib_share"] - 0.5).abs()
    merged["extremity_gap"] = merged["model_extremity"] - merged["ces_extremity"]
    merged.to_csv(ROBUST / "composition_per_issue.csv", index=False)

    # Per-group summary
    recs = []
    for (model, fam, grp), sub in merged.groupby(["model", "cue_family", "cue_group"]):
        if len(sub) < 3:
            continue
        x = sub["ces_lib_share"].to_numpy()
        y = sub["model_lib_share"].to_numpy()
        slope = np.polyfit(x, y, 1)[0] if np.ptp(x) > 1e-6 else np.nan
        recs.append({
            "model": model, "cue_family": fam, "cue_group": grp,
            "n_issues": len(sub),
            "collapse_index": float(sub["extremity_gap"].mean()),
            "collapse_index_se": float(sub["extremity_gap"].std(ddof=1) / np.sqrt(len(sub))),
            "frac_issues_more_extreme": float((sub["extremity_gap"] > 0).mean()),
            "mean_model_extremity": float(sub["model_extremity"].mean()),
            "mean_ces_extremity": float(sub["ces_extremity"].mean()),
            "composition_slope": slope,
            "mean_neutral_rate": float(sub["neutral_rate"].mean()),
        })
    summary = pd.DataFrame(recs)
    summary.to_csv(ROBUST / "composition_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== Composition-level flattening (variance collapse within cue group) ===")
    print("collapse_index = mean(model extremity - CES extremity); >0 => model more one-sided\n")
    show = summary[summary.cue_family.isin(["explicit_political", "explicit_demographic",
                                            "implicit_demographic", "baseline"])]
    for m in MODELS:
        print(f"--- {MODEL_LABEL[m]} ---")
        s = summary[summary.model == m].sort_values("collapse_index", ascending=False)
        for _, r in s.iterrows():
            flag = "COLLAPSE" if r["collapse_index"] > 2 * r["collapse_index_se"] else "        "
            print(f"  {r['cue_family']:>20} {r['cue_group']:<12} "
                  f"collapse={r['collapse_index']:+.3f}±{r['collapse_index_se']:.3f} "
                  f"[{flag}]  mExtr={r['mean_model_extremity']:.2f} cExtr={r['mean_ces_extremity']:.2f} "
                  f"slope={r['composition_slope']:+.2f}")
        print()

    # Headline: overall model vs CES extremity, and how many cells collapse
    n_collapse = (summary["collapse_index"] > 2 * summary["collapse_index_se"]).sum()
    print(f"Cells with significant variance collapse: {n_collapse}/{len(summary)}")
    print(f"Mean collapse index across all cells: {summary['collapse_index'].mean():+.3f}")
    print(f"Wrote {ROBUST/'composition_summary.csv'} and composition_per_issue.csv")
    make_figure(merged, summary)


def make_figure(merged, summary):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    focus = [("explicit_political", "democrat", "Democrat (label)"),
             ("explicit_political", "republican", "Republican (label)"),
             ("explicit_demographic", "black_woman", "Black woman (label)"),
             ("implicit_demographic", "black_woman", "“Black-female” name")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharex=True, sharey=True)
    for ax, (fam, grp, title) in zip(axes, focus):
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        for m in MODELS:
            sub = merged[(merged.model == m) & (merged.cue_family == fam) & (merged.cue_group == grp)]
            ax.scatter(sub["ces_lib_share"], sub["model_lib_share"], s=28,
                       color=MODEL_COLOUR[m], alpha=0.8, label=MODEL_LABEL[m], edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.axhline(0.5, color="#ddd", lw=0.6)
        ax.axvline(0.5, color="#ddd", lw=0.6)
        ax.set_xlabel("CES liberal share (real group)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Model liberal share\n(among directional responses)")
    axes[-1].legend(fontsize=8, frameon=True, loc="lower right")
    fig.suptitle("Composition, not just means: model directional share vs real group share, per issue.  "
                 "Points pushed to 0/1 = within-group variance collapse.", y=1.02, fontsize=11)
    fig.tight_layout()
    p = Path("figures/robustness/composition_flattening.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
