#!/usr/bin/env python3
"""Analysis #1: does name legibility predict the generation-side stance shift?

Merges, per name, the cue-probe legibility (how reliably the model perceives the
intended race from the name) with the stance shift that same name induces in the
generation experiment. The question is whether the small name-cue stance effect
is a *legibility* failure (the model never reads the race) or an *action* failure
(it reads the race but does not act on it).

Two legibility measures are reported because they answer different versions of
the question:

  - ecological recall  (--probe = cue_probe_questions.csv): the name sits inside a
    real writing request, matching the generation prompt structure. This is the
    legibility that is causally relevant to the stance shift.
  - name-only recall   (--probe = cue_probe.csv): the bare "My name is X." cue,
    the upper bound on what the name alone can signal.

Stance comes from stance_by_name.csv (built from evaluated_with_effects.csv):
per-name mean ``cue_effect`` (shift from the no-cue baseline) on the liberal axis.

Outputs a merged table, a Pearson/Spearman correlation (overall and within the
Black subgroups, where there is variation to explain), and a scatter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SUBGROUP_ORDER = ["white_man", "white_woman", "black_man", "black_woman"]
COLORS = {"white_man": "#39c", "white_woman": "#9cf", "black_man": "#a40", "black_woman": "#f93"}


def _probe_recall(probe_csv: Path) -> pd.DataFrame:
    """Per-name race recall + abstain from a cue-probe CSV."""
    d = pd.read_csv(probe_csv)
    r = d[d["attribute"] == "race"].copy()
    r["recall"] = r["parsed_value"] == r["intended_race"]
    r["abstain"] = r["parsed_value"] == "cannot_tell"
    r["name_l"] = r["name"].str.lower()
    return r.groupby(["subgroup", "name_l"]).agg(
        race_recall=("recall", "mean"), race_abstain=("abstain", "mean"), probe_n=("recall", "size")
    ).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, help="cue_probe_questions.csv (ecological) or cue_probe.csv (name-only)")
    ap.add_argument("--stance", default="results/main/stance_by_name.csv")
    ap.add_argument("--out-dir", default="results/legibility_vs_stance")
    ap.add_argument("--figures-dir", default="figures/legibility_vs_stance")
    ap.add_argument("--label", default="ecological", help="tag for output filenames (e.g. ecological / name_only)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    stance = pd.read_csv(args.stance)
    stance["name_l"] = stance["name"].str.lower()
    recall = _probe_recall(Path(args.probe))

    merged = stance.merge(recall, on=["subgroup", "name_l"], how="inner")
    if merged.empty:
        print("No overlapping names between stance and probe — probe the generation "
              "names first (data/input/names/names_generation.csv).")
        return 1
    merged = merged.sort_values(["subgroup", "race_recall"])
    merged.to_csv(out_dir / f"merged_{args.label}.csv", index=False)

    def corr(df, x="race_recall", y="shift"):
        if len(df) < 3 or df[x].nunique() < 2:
            return None
        pr = stats.pearsonr(df[x], df[y]); sr = stats.spearmanr(df[x], df[y])
        return pr.statistic, pr.pvalue, sr.statistic, sr.pvalue, len(df)

    print(f"=== merged {len(merged)} names ({args.label} legibility) ===")
    print(merged[["subgroup", "name", "race_recall", "race_abstain", "shift", "stance", "n"]]
          .round(3).to_string(index=False))
    overall = corr(merged)
    black = corr(merged[merged.subgroup.str.startswith("black")])
    print("\nrace_recall vs stance shift:")
    if overall:
        print(f"  overall (n={overall[4]}): Pearson r={overall[0]:.3f} p={overall[1]:.3f} | "
              f"Spearman rho={overall[2]:.3f} p={overall[3]:.3f}")
    if black:
        print(f"  Black names (n={black[4]}): Pearson r={black[0]:.3f} p={black[1]:.3f} | "
              f"Spearman rho={black[2]:.3f} p={black[3]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for sg in SUBGROUP_ORDER:
        s = merged[merged.subgroup == sg]
        if len(s):
            ax.scatter(s["race_recall"], s["shift"], label=sg, color=COLORS[sg], s=60)
            for _, row in s.iterrows():
                ax.annotate(row["name"], (row["race_recall"], row["shift"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(f"race legibility — {args.label} recall (model perceives intended race)")
    ax.set_ylabel("stance shift from baseline (liberal axis)")
    ax.set_title("Does name legibility predict the stance shift?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"legibility_vs_stance_{args.label}.png", dpi=150)
    print(f"\nWrote merge to {out_dir} and scatter to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
