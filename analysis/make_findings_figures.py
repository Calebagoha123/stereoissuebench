#!/usr/bin/env python3
"""Presentation figures for the PCT arm and the cue-legibility probe.

Reads the small summary tables the two report scripts already wrote and renders
clean, self-contained figures into ``--figures-dir``. Each figure is skipped if
its input table is missing, so this runs against whatever you have locally.

Inputs (defaults):
  results/pct_all_paper_v2/pct_cue_effects.csv      (framing-matched, all families)
  results/pct_all_paper_v2/pct_scores_by_condition.csv
  results/pct_names_full/pct_cue_effects.csv        (~140 names/subgroup)
  results/cue_probe/legibility_by_subgroup.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

FAMILY_ORDER = [
    "explicit_political", "implicit_political",
    "explicit_demographic", "implicit_demographic",
]
FAMILY_COLOURS = {
    "explicit_political": "#c0392b",
    "implicit_political": "#e08aa8",
    "explicit_demographic": "#2e6da4",
    "implicit_demographic": "#27915b",
}
FAMILY_LABELS = {
    "explicit_political": "Explicit political  (As a … Republican)",
    "implicit_political": "Implicit political  (… lives in Texas)",
    "explicit_demographic": "Explicit demographic  (As a Black man)",
    "implicit_demographic": "Implicit demographic  (named Jamal)",
}
SUBGROUP_LABEL = {
    "white_man": "White man", "white_woman": "White woman",
    "black_man": "Black man", "black_woman": "Black woman",
}


def _short(condition: str) -> str:
    return condition.split("/", 1)[-1] if "/" in condition else condition


def fig_cue_effects(effects_csv: Path, scores_csv: Path, out: Path) -> None:
    """Headline forest plot: cue effect on PCT lean vs the no-cue baseline."""
    eff = pd.read_csv(effects_csv)
    eff = eff[eff["axis"] == "overall"].copy()
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    eff["frank"] = eff["cue_family"].map(rank).fillna(99)
    eff = eff.sort_values(["frank", "effect"], ascending=[False, True]).reset_index(drop=True)

    baseline = ""
    if scores_csv.exists():
        s = pd.read_csv(scores_csv).set_index("condition")
        if "baseline" in s.index:
            baseline = f"  (baseline = {s.loc['baseline', 'overall_mean']:+.2f} liberal)"

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axvspan(-0.05, 0.05, color="0.85", alpha=0.5, zorder=0, label="≈ no shift")
    ax.axvline(0, color="0.4", lw=1)
    for y, row in eff.iterrows():
        c = FAMILY_COLOURS.get(row["cue_family"], "#333")
        ax.errorbar(row["effect"], y, xerr=[[row["effect"] - row["lo"]], [row["hi"] - row["effect"]]],
                    fmt="o", color=c, ecolor=c, capsize=3, markersize=7, lw=1.6)
    ax.set_yticks(range(len(eff)))
    ax.set_yticklabels([_short(c) for c in eff["condition"]])
    ax.set_xlabel("Shift in Political-Compass lean vs no-cue baseline\n"
                  "(− = more conservative · + = more liberal)")
    ax.set_title(f"Only explicit political identity moves Qwen's compass{baseline}", fontsize=12)
    handles = [plt.Line2D([0], [0], marker="o", color=FAMILY_COLOURS[f], lw=0, label=FAMILY_LABELS[f])
               for f in FAMILY_ORDER if f in set(eff["cue_family"])]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_names_robustness(small_csv: Path, full_csv: Path, out: Path) -> None:
    """3-names-per-cell vs ~140-names-per-cell: the apparent name effect collapses."""
    def load(path):
        d = pd.read_csv(path)
        d = d[(d["axis"] == "overall") & (d["cue_family"] == "implicit_demographic")].copy()
        d["subgroup"] = d["condition"].map(_short)
        return d.set_index("subgroup")

    small, full = load(small_csv), load(full_csv)
    subgroups = [s for s in SUBGROUP_LABEL if s in full.index]
    y = range(len(subgroups))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axvspan(-0.05, 0.05, color="0.85", alpha=0.5, zorder=0)
    ax.axvline(0, color="0.4", lw=1)
    for i, sg in enumerate(subgroups):
        if sg in small.index:
            r = small.loc[sg]
            ax.errorbar(r["effect"], i + 0.16, xerr=[[r["effect"] - r["lo"]], [r["hi"] - r["effect"]]],
                        fmt="o", mfc="white", mec="#999", ecolor="#999", capsize=3, markersize=7,
                        label="3 names / subgroup" if i == 0 else None)
        r = full.loc[sg]
        ax.errorbar(r["effect"], i - 0.16, xerr=[[r["effect"] - r["lo"]], [r["hi"] - r["effect"]]],
                    fmt="o", color="#27915b", ecolor="#27915b", capsize=3, markersize=8,
                    label="~140 names / subgroup" if i == 0 else None)
    ax.set_yticks(list(y))
    ax.set_yticklabels([SUBGROUP_LABEL[s] for s in subgroups])
    ax.set_xlabel("Name-cue effect on PCT lean vs baseline  (− = more conservative)")
    ax.set_title("Name-cue effects vanish once you average over real name variation", fontsize=12)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_probe_legibility(legib_csv: Path, out: Path) -> None:
    """Name legibility: race recall (White default), gender recall, political flatline."""
    d = pd.read_csv(legib_csv).set_index("subgroup")
    subgroups = [s for s in SUBGROUP_LABEL if s in d.index]
    x = range(len(subgroups))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], d.loc[subgroups, "race_recall"], w, color="#2e6da4", label="race recall")
    ax.bar([i + w / 2 for i in x], d.loc[subgroups, "gender_recall"], w, color="#27915b", label="gender recall")
    if "political_mean" in d.columns:
        ax.plot(list(x), d.loc[subgroups, "political_mean"], "D", color="#c0392b",
                markersize=8, label="inferred political lean")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SUBGROUP_LABEL[s] for s in subgroups])
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("rate  /  inferred lean (−1…+1)")
    ax.set_title("Qwen reads gender and White names, defaults Black names to White,\n"
                 "and infers no political lean from any name", fontsize=12)
    ax.legend(fontsize=9, loc="center right", framealpha=0.9)
    for i, sg in enumerate(subgroups):
        ax.annotate(f"{d.loc[sg, 'race_recall']:.0%}", (i - w / 2, d.loc[sg, "race_recall"] + 0.02),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pct-all", default="results/pct_all_paper_v2")
    p.add_argument("--pct-names-full", default="results/pct_names_full")
    p.add_argument("--probe", default="results/cue_probe")
    p.add_argument("--figures-dir", default="figures/findings")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    out = Path(a.figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    pct_all, pct_full, probe = Path(a.pct_all), Path(a.pct_names_full), Path(a.probe)

    if (pct_all / "pct_cue_effects.csv").exists():
        fig_cue_effects(pct_all / "pct_cue_effects.csv",
                        pct_all / "pct_scores_by_condition.csv",
                        out / "pct_cue_effects.png")
    if (pct_all / "pct_cue_effects.csv").exists() and (pct_full / "pct_cue_effects.csv").exists():
        fig_names_robustness(pct_all / "pct_cue_effects.csv",
                             pct_full / "pct_cue_effects.csv",
                             out / "pct_names_robustness.png")
    if (probe / "legibility_by_subgroup.csv").exists():
        fig_probe_legibility(probe / "legibility_by_subgroup.csv", out / "probe_legibility.png")
    print(f"\nFigures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
