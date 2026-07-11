#!/usr/bin/env python3
"""Presentation figures for the PCT arm.

Reads the small summary tables ``pct_report.py`` already wrote and renders clean,
self-contained figures into ``--figures-dir``. Each figure is skipped if its
input table is missing, so this runs against whatever you have locally. (The
cue-legibility probe is reported as a table in docs/findings.md, not here.)

Inputs (defaults):
  results/pct_all_paper_v2/pct_cue_effects.csv   (framing-matched, all families)
  results/pct_names_full/pct_cue_effects.csv     (~140 names/subgroup)
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


def fig_cue_effects(effects_csv: Path, out: Path) -> None:
    """Headline forest plot: cue effect on PCT lean vs the no-cue baseline."""
    eff = pd.read_csv(effects_csv)
    eff = eff[eff["axis"] == "overall"].copy()
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    eff["frank"] = eff["cue_family"].map(rank).fillna(99)
    eff = eff.sort_values(["frank", "effect"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axvspan(-0.05, 0.05, color="0.85", alpha=0.5, zorder=0, label="negligible (±0.05)")
    ax.axvline(0, color="0.4", lw=1.2)
    ax.annotate("no-cue baseline", xy=(0, len(eff) - 0.5), ha="center", va="bottom",
                fontsize=8.5, color="0.35")
    for y, row in eff.iterrows():
        c = FAMILY_COLOURS.get(row["cue_family"], "#333")
        ax.errorbar(row["effect"], y, xerr=[[row["effect"] - row["lo"]], [row["hi"] - row["effect"]]],
                    fmt="o", color=c, ecolor=c, capsize=3, markersize=7, lw=1.6)
    ax.set_yticks(range(len(eff)))
    ax.set_yticklabels([_short(c) for c in eff["condition"]])
    ax.set_xlabel("Shift in Political-Compass lean vs no-cue baseline\n"
                  "(− = more conservative · + = more liberal)")
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pct-all", default="results/pct_all_paper_v2")
    p.add_argument("--pct-names-full", default="results/pct_names_full")
    p.add_argument("--figures-dir", default="figures/findings")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    out = Path(a.figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    pct_all, pct_full = Path(a.pct_all), Path(a.pct_names_full)

    if (pct_all / "pct_cue_effects.csv").exists():
        fig_cue_effects(pct_all / "pct_cue_effects.csv", out / "pct_cue_effects.png")
    if (pct_all / "pct_cue_effects.csv").exists() and (pct_full / "pct_cue_effects.csv").exists():
        fig_names_robustness(pct_all / "pct_cue_effects.csv",
                             pct_full / "pct_cue_effects.csv",
                             out / "pct_names_robustness.png")
    print(f"\nFigures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
