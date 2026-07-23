#!/usr/bin/env python3
"""Implicit-cue instance-level breakdown (check E): per-name and per-state effects
behind the group estimates.

Defends the names-null against "your null is an artifact of unlucky name
selection" (Tonneau et al.): if every name in a group clusters at zero, the null
is about names as a class; if a couple of names carry an effect and the rest do
not, the story changes. Also substantiates the bootstrap-over-instances claim that
the group intervals reflect group-level (not instance-level) uncertainty.

For each Arm-B group we compute every instance's mean liberal score minus the
model's baseline, and summarise the within-group spread. Reads
results/full_3x/bert_eval_*.csv. Writes results/robustness/instance_effects.csv
and a strip/forest figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, MODEL_COLOUR, ROBUST, load_model

ARM_B = [
    ("implicit_demographic", "black_woman"), ("implicit_demographic", "black_man"),
    ("implicit_demographic", "white_woman"), ("implicit_demographic", "white_man"),
    ("implicit_political", "blue_state"), ("implicit_political", "red_state"),
    ("implicit_political", "swing_state"),
]


def main():
    recs = []
    for m in MODELS:
        d = load_model(m)
        base_all = d[d.cue_family == "baseline"]
        for fam, grp in ARM_B:
            cue = d[(d.cue_family == fam) & (d.cue_group == grp)]
            if cue.empty:
                continue
            base = base_all[base_all.template_id.isin(cue.template_id.unique())]
            base_mean = base["y"].mean()
            for inst, g in cue.groupby("instance"):
                recs.append({"model": m, "cue_family": fam, "cue_group": grp,
                             "instance": inst, "n": len(g),
                             "effect": g["y"].mean() - base_mean})
    df = pd.DataFrame(recs)
    df.to_csv(ROBUST / "instance_effects.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== Instance-level effects behind the implicit-cue group estimates ===\n")
    summ = (df.groupby(["model", "cue_family", "cue_group"])
            .agg(n_instances=("instance", "nunique"),
                 mean_effect=("effect", "mean"),
                 sd_across_instances=("effect", "std"),
                 min_effect=("effect", "min"), max_effect=("effect", "max")).reset_index())
    for m in MODELS:
        print(f"--- {MODEL_LABEL[m]} ---")
        for _, r in summ[summ.model == m].iterrows():
            print(f"  {r['cue_group']:>12} ({r['n_instances']:2d} instances): "
                  f"group mean={r['mean_effect']:+.3f}  within-group SD={r['sd_across_instances']:.3f}  "
                  f"range=[{r['min_effect']:+.3f},{r['max_effect']:+.3f}]")
        print()
    # names-null specific: does any single name carry an outlier effect?
    names = df[df.cue_family == "implicit_demographic"]
    print("Name cues: fraction of individual names with |effect| > 0.10:")
    print(f"  {(names['effect'].abs() > 0.10).mean()*100:.1f}%  "
          f"(if the null were driven by a few strong names this would be high)")
    print(f"  within-group instance SD (median across name groups): "
          f"{summ[summ.cue_family=='implicit_demographic']['sd_across_instances'].median():.3f}")
    make_figure(df)
    print(f"\nWrote {ROBUST/'instance_effects.csv'}")


def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    groups = [("implicit_demographic", "black_woman", "“Black-female” names"),
              ("implicit_demographic", "black_man", "“Black-male” names"),
              ("implicit_demographic", "white_woman", "“White-female” names"),
              ("implicit_demographic", "white_man", "“White-male” names"),
              ("implicit_political", "blue_state", "Blue states"),
              ("implicit_political", "red_state", "Red states"),
              ("implicit_political", "swing_state", "Swing states")]
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ypos = list(range(len(groups)))[::-1]
    ax.axvline(0, color="#888", lw=1, ls="--")
    for y, (fam, grp, lab) in zip(ypos, groups):
        for m in MODELS:
            sub = df[(df.model == m) & (df.cue_family == fam) & (df.cue_group == grp)]
            jit = (MODELS.index(m) - 1) * 0.18
            ax.scatter(sub["effect"], [y + jit] * len(sub), s=22, alpha=0.6,
                       color=MODEL_COLOUR[m], edgecolor="white", linewidth=0.3,
                       label=MODEL_LABEL[m] if y == ypos[0] else None)
            ax.scatter([sub["effect"].mean()], [y + jit], s=90, marker="|",
                       color=MODEL_COLOUR[m], linewidth=2.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for *_, lab in groups])
    ax.set_xlabel("Instance effect = instance mean − baseline (liberal score)")
    ax.set_xlim(-0.35, 0.35)
    ax.legend(fontsize=8.5, frameon=True, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Every name / state behind the group estimate (thick tick = group mean)", fontsize=11)
    fig.tight_layout()
    p = Path("figures/robustness/instance_breakdown.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
