#!/usr/bin/env python3
"""Summarise the Political Compass Test arm (pipeline/06_run_pct.py).

Works for either cue set the runner produces (``--cue-set names`` or ``all``).
A "condition" is keyed by ``cue_family`` + ``cue_group`` jointly, because the
explicit-demographic and implicit-demographic (name) cues SHARE a cue_group
("white_man" etc.) — "I am a White man." and "My name is Brad." must not be
merged. ``baseline`` is the single no-cue condition.

For each condition it reports:
  - overall PCT lean on the pipeline's [-1, +1] ``liberal_score`` axis
    (+1 liberal/left, -1 conservative/right), pooled over coded items;
  - the same split by PCT axis (economic / social) — a transparent 2-D position;
  - the cue effect: condition lean minus baseline, paired within PCT item, with
    an item-clustered bootstrap CI.

The explicit-political cues (Democrat / Republican) are the manipulation check:
following Tornberg & Schimmel (arXiv:2604.27633), an explicit conservative cue
should swing the PCT sharply right while the progressive cue barely moves it
(left ceiling). A near-zero effect from the implicit name cues is interpretable
only against that contrast.

Note on the 2-D plot: politicalcompass.org's coordinates use proprietary
nonlinear weights we do not have; the axes here are our transparent per-axis
mean ``liberal_score``. Ambiguous items (ideo_direction == 0) carry no partisan
signal and are excluded from the lean; their letters remain in the raw file.
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
AXES = ["economic", "social"]

# Family display order and colours; conditions sort by family then group.
FAMILY_ORDER = [
    "baseline",
    "explicit_political",
    "implicit_political",
    "explicit_demographic",
    "implicit_demographic",
]
FAMILY_COLOURS = {
    "baseline": "#444",
    "explicit_political": "#c33",
    "implicit_political": "#e8a",
    "explicit_demographic": "#39c",
    "implicit_demographic": "#3a6",
}


def _add_condition(frame: pd.DataFrame) -> pd.DataFrame:
    """Condition key = cue_family/cue_group (baseline kept as just 'baseline')."""
    frame = frame.copy()
    frame["condition"] = np.where(
        frame["cue_family"] == "baseline",
        BASELINE,
        frame["cue_family"].astype(str) + "/" + frame["cue_group"].astype(str),
    )
    return frame


def _condition_order(frame: pd.DataFrame) -> list[str]:
    fam = frame.groupby("condition")["cue_family"].first()
    grp = frame.groupby("condition")["cue_group"].first()
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(fam.index, key=lambda c: (rank.get(fam[c], 99), grp[c]))


def _item_means(coded: pd.DataFrame, axis: str | None = None) -> pd.DataFrame:
    """Per (condition, pct_id) mean liberal_score, optionally one PCT axis.

    Averaging over realizations-in-condition and repeats first gives one value
    per item per condition, so conditions are paired by item against baseline."""
    frame = coded if axis is None else coded[coded["axis"] == axis]
    return frame.groupby(["condition", "pct_id"])["liberal_score"].mean().reset_index()


def _condition_summary(
    coded: pd.DataFrame, conditions: list[str], n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    fam = coded.groupby("condition")["cue_family"].first()
    records = []
    for condition in conditions:
        row: dict[str, object] = {"condition": condition, "cue_family": fam.get(condition, "")}
        for axis_name, axis in [("overall", None)] + [(a, a) for a in AXES]:
            per_item = _item_means(coded, axis)
            series = per_item[per_item["condition"] == condition].set_index("pct_id")[
                "liberal_score"
            ]
            if series.empty:
                row[f"{axis_name}_mean"] = row[f"{axis_name}_lo"] = row[f"{axis_name}_hi"] = np.nan
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


def _cue_effects(
    coded: pd.DataFrame, conditions: list[str], n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    fam = coded.groupby("condition")["cue_family"].first()
    records = []
    for axis_name, axis in [("overall", None)] + [(a, a) for a in AXES]:
        per_item = _item_means(coded, axis)
        wide = per_item.pivot(index="pct_id", columns="condition", values="liberal_score")
        if BASELINE not in wide.columns:
            continue
        for condition in conditions:
            if condition == BASELINE or condition not in wide.columns:
                continue
            paired = wide[[BASELINE, condition]].dropna()
            if paired.empty:
                continue
            diff = (paired[condition] - paired[BASELINE]).to_numpy()
            boot = np.array(
                [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
            )
            records.append(
                {
                    "condition": condition,
                    "cue_family": fam.get(condition, ""),
                    "axis": axis_name,
                    "effect": float(diff.mean()),
                    "lo": float(np.percentile(boot, 2.5)),
                    "hi": float(np.percentile(boot, 97.5)),
                    "n_items": int(len(diff)),
                }
            )
    return pd.DataFrame(records)


def _plot_compass(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for cond, srow in summary.iterrows():
        x, y = srow["economic_mean"], srow["social_mean"]
        if np.isnan(x) or np.isnan(y):
            continue
        fam = srow["cue_family"]
        ax.scatter(x, y, s=90, color=FAMILY_COLOURS.get(fam, "#000"), zorder=3,
                   marker="*" if cond == BASELINE else "o")
        label = cond.split("/", 1)[-1] if "/" in cond else cond
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("economic lean  (-1 right … +1 left/liberal)")
    ax.set_ylabel("social lean  (-1 authoritarian … +1 progressive)")
    ax.set_title("PCT position by cue condition (liberal-positive axes)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f)
        for f, c in FAMILY_COLOURS.items() if f in summary["cue_family"].values
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cue_effects(effects: pd.DataFrame, out_path: Path) -> None:
    overall = effects[effects["axis"] == "overall"].copy()
    if overall.empty:
        return
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    overall = overall.sort_values(
        by=["cue_family", "effect"], key=lambda s: s.map(rank) if s.name == "cue_family" else s
    )
    labels = [c.split("/", 1)[-1] if "/" in c else c for c in overall["condition"]]
    y = np.arange(len(overall))
    means = overall["effect"].to_numpy()
    lo, hi = overall["lo"].to_numpy(), overall["hi"].to_numpy()
    colours = [FAMILY_COLOURS.get(f, "#000") for f in overall["cue_family"]]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(overall) + 1)))
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="none", ecolor="#999", capsize=3, zorder=1)
    ax.scatter(means, y, color=colours, zorder=2)
    ax.axvline(0.0, color="grey", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("PCT lean shift vs no-cue baseline  (+ = more liberal)")
    ax.set_title("Cue effect on PCT lean, by condition")
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
    frame = _add_condition(frame)
    parsed = frame["letter"].isin(list("ABCD")).sum()
    print(f"Loaded {len(frame)} PCT rows; parse rate {parsed}/{len(frame)} "
          f"({parsed / len(frame) * 100:.1f}%).")

    coded = frame.dropna(subset=["liberal_score"]).copy()
    conditions = _condition_order(coded)
    rng = np.random.default_rng(args.seed)

    summary = _condition_summary(coded, conditions, args.bootstrap, rng)
    effects = _cue_effects(coded, conditions, args.bootstrap, rng)
    per_value = (
        coded.groupby(["cue_family", "cue_group", "cue_value"])["liberal_score"]
        .mean().reset_index()
    )

    summary.round(4).to_csv(out_dir / "pct_scores_by_condition.csv")
    effects.round(4).to_csv(out_dir / "pct_cue_effects.csv", index=False)
    per_value.round(4).to_csv(out_dir / "pct_by_cue_value.csv", index=False)

    _plot_compass(summary, fig_dir / "pct_compass.png")
    _plot_cue_effects(effects, fig_dir / "pct_cue_effects.png")

    print("\nPCT lean by condition (+ = liberal/left):\n", summary.round(3).to_string())
    if not effects.empty:
        print("\nCue effects vs baseline (overall axis):\n",
              effects[effects["axis"] == "overall"].round(3).to_string(index=False))
    print(f"\nWrote tables to {out_dir} and figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
