#!/usr/bin/env python3
"""Summarise the Political Compass Test arm (pipeline/06_run_pct.py).

Reports, for the no-cue baseline and each implicit-demographic name subgroup:

  - overall PCT lean on the pipeline's [-1, +1] ``liberal_score`` axis
    (+1 liberal / left, -1 conservative / right), pooled over coded items;
  - the same split by PCT axis (economic / social) — a transparent 2-D position;
  - the cue effect: subgroup lean minus baseline lean, paired within PCT item,
    with an item-clustered bootstrap CI.

Note on the 2-D plot: politicalcompass.org's published coordinates use
proprietary nonlinear weights we do not have. We therefore do NOT claim to
reproduce those coordinates; the axes here are the transparent mean
``liberal_score`` per PCT axis under our own coding, on the same liberal scale as
the rest of the report. Ambiguous items (ideo_direction == 0) carry no partisan
signal and are excluded from the lean; their letters are still in the raw file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE = "baseline"
SUBGROUP_ORDER = ["white_man", "white_woman", "black_man", "black_woman"]
CONDITION_ORDER = [BASELINE] + SUBGROUP_ORDER
AXES = ["economic", "social"]


def _item_means(coded: pd.DataFrame, axis: str | None = None) -> pd.DataFrame:
    """Per (cue_group, pct_id) mean liberal_score, optionally one PCT axis.

    Averaging over names-in-subgroup and repeats first gives one value per item
    per condition, so baseline and subgroup conditions are paired by item."""
    frame = coded if axis is None else coded[coded["axis"] == axis]
    return (
        frame.groupby(["cue_group", "pct_id"])["liberal_score"].mean().reset_index()
    )


def _condition_summary(coded: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Mean lean (overall + per axis) per condition, item-clustered bootstrap CI."""
    records = []
    for cue_group in CONDITION_ORDER:
        row: dict[str, object] = {"condition": cue_group}
        for axis_name, axis in [("overall", None)] + [(a, a) for a in AXES]:
            per_item = _item_means(coded, axis)
            series = per_item[per_item["cue_group"] == cue_group].set_index("pct_id")[
                "liberal_score"
            ]
            if series.empty:
                row[f"{axis_name}_mean"] = np.nan
                row[f"{axis_name}_lo"] = np.nan
                row[f"{axis_name}_hi"] = np.nan
                continue
            items = series.index.to_numpy()
            boot = np.array(
                [series.loc[rng.choice(items, size=len(items), replace=True)].mean()
                 for _ in range(n_boot)]
            )
            row[f"{axis_name}_mean"] = float(series.mean())
            row[f"{axis_name}_lo"] = float(np.percentile(boot, 2.5))
            row[f"{axis_name}_hi"] = float(np.percentile(boot, 97.5))
            if axis_name == "overall":
                row["n_items"] = int(series.size)
        records.append(row)
    return pd.DataFrame(records).set_index("condition")


def _cue_effects(coded: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Subgroup lean minus baseline, paired within PCT item, bootstrap CI over items."""
    records = []
    for axis_name, axis in [("overall", None)] + [(a, a) for a in AXES]:
        per_item = _item_means(coded, axis)
        wide = per_item.pivot(index="pct_id", columns="cue_group", values="liberal_score")
        if BASELINE not in wide.columns:
            continue
        for subgroup in SUBGROUP_ORDER:
            if subgroup not in wide.columns:
                continue
            paired = wide[[BASELINE, subgroup]].dropna()
            if paired.empty:
                continue
            diff = (paired[subgroup] - paired[BASELINE]).to_numpy()
            boot = np.array(
                [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
            )
            records.append(
                {
                    "subgroup": subgroup,
                    "axis": axis_name,
                    "effect": float(diff.mean()),
                    "lo": float(np.percentile(boot, 2.5)),
                    "hi": float(np.percentile(boot, 97.5)),
                    "n_items": int(len(diff)),
                }
            )
    return pd.DataFrame(records)


def _plot_compass(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colours = {
        BASELINE: "#444", "white_man": "#39c", "white_woman": "#3b6",
        "black_man": "#825", "black_woman": "#c63",
    }
    for cond in CONDITION_ORDER:
        if cond not in summary.index:
            continue
        x = summary.loc[cond, "economic_mean"]
        y = summary.loc[cond, "social_mean"]
        ax.scatter(x, y, s=90, color=colours.get(cond, "#000"), zorder=3,
                   marker="*" if cond == BASELINE else "o")
        ax.annotate(cond, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("economic lean  (-1 right … +1 left/liberal)")
    ax.set_ylabel("social lean  (-1 authoritarian … +1 progressive)")
    ax.set_title("PCT position by cue condition (liberal-positive axes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cue_effects(effects: pd.DataFrame, out_path: Path) -> None:
    overall = effects[effects["axis"] == "overall"].set_index("subgroup")
    subgroups = [s for s in SUBGROUP_ORDER if s in overall.index]
    if not subgroups:
        return
    y = np.arange(len(subgroups))
    means = overall.loc[subgroups, "effect"].to_numpy()
    lo = overall.loc[subgroups, "lo"].to_numpy()
    hi = overall.loc[subgroups, "hi"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="o", color="#825", capsize=4)
    ax.axvline(0.0, color="grey", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(subgroups)
    ax.set_xlabel("PCT lean shift vs no-cue baseline  (+ = more liberal)")
    ax.set_title("Name-cue effect on PCT lean, by subgroup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pct", required=True, help="pct.csv from stage 06")
    parser.add_argument("--out-dir", default="results/pct")
    parser.add_argument("--figures-dir", default="figures/pct")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.pct)
    frame["liberal_score"] = pd.to_numeric(frame["liberal_score"], errors="coerce")
    parsed = frame["letter"].isin(list("ABCD")).sum()
    print(f"Loaded {len(frame)} PCT rows; parse rate {parsed}/{len(frame)} "
          f"({parsed / len(frame) * 100:.1f}%).")

    # Coded items only (ambiguous items have no liberal_score).
    coded = frame.dropna(subset=["liberal_score"]).copy()
    rng = np.random.default_rng(args.seed)

    summary = _condition_summary(coded, args.bootstrap, rng)
    effects = _cue_effects(coded, args.bootstrap, rng)

    per_name = (
        coded.groupby(["cue_group", "cue_value"])["liberal_score"].mean()
        .reset_index().rename(columns={"cue_value": "name"})
    )

    summary.round(4).to_csv(out_dir / "pct_scores_by_condition.csv")
    effects.round(4).to_csv(out_dir / "pct_cue_effects.csv", index=False)
    per_name.round(4).to_csv(out_dir / "pct_by_name.csv", index=False)

    _plot_compass(summary, fig_dir / "pct_compass.png")
    _plot_cue_effects(effects, fig_dir / "pct_cue_effects.png")

    print("\nPCT lean by condition (+ = liberal/left):\n", summary.round(3).to_string())
    print("\nCue effects vs baseline:\n", effects.round(3).to_string(index=False))
    print(f"\nWrote tables to {out_dir} and figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
