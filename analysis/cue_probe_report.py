#!/usr/bin/env python3
"""Summarise the cue-legibility probe (pipeline/05_run_cue_probe.py).

Reads the per-(name, attribute, repeat) probe rows and reports, per race-gender
subgroup:

  - race / gender RECALL of the intended subgroup (the Tonneau legibility metric)
  - the "Cannot tell" ABSTENTION rate (how often the cue is judged uninformative)
  - the mean inferred POLITICAL leaning on the [-1, +1] liberal scale, with a
    name-clustered bootstrap CI, so it sits on the same axis as the generation
    -side stance shift and the CES subgroup mean.

Outputs tables to --out-dir and figures to --figures-dir. The inferred-lean
series is written so it can be overlaid on make_report_figures.py's
model-vs-CES panels for the three-way (inferred / expressed / actual) comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUBGROUP_ORDER = ["white_man", "white_woman", "black_man", "black_woman"]
ABSTAIN = "cannot_tell"
PARSE_ERROR = "PARSE_ERROR"


def _recall_table(frame: pd.DataFrame, attribute: str, intended_col: str) -> pd.DataFrame:
    rows = frame[frame["attribute"] == attribute].copy()
    rows["correct"] = rows["parsed_value"] == rows[intended_col]
    rows["abstain"] = rows["parsed_value"] == ABSTAIN
    rows["parse_error"] = rows["parsed_value"] == PARSE_ERROR
    grouped = rows.groupby("subgroup").agg(
        recall=("correct", "mean"),
        abstain=("abstain", "mean"),
        parse_error=("parse_error", "mean"),
        n=("correct", "size"),
    )
    return grouped.rename(columns={c: f"{attribute}_{c}" for c in grouped.columns})


def _confusion(frame: pd.DataFrame, attribute: str, intended_col: str) -> pd.DataFrame:
    rows = frame[frame["attribute"] == attribute]
    return pd.crosstab(rows[intended_col], rows["parsed_value"], normalize="index").round(3)


def _political_summary(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = frame[frame["attribute"] == "political"].copy()
    rows["value"] = pd.to_numeric(rows["parsed_value"], errors="coerce")
    rows = rows.dropna(subset=["value"])
    rng = np.random.default_rng(seed)

    records = []
    for subgroup in SUBGROUP_ORDER:
        sub = rows[rows["subgroup"] == subgroup]
        if sub.empty:
            continue
        # Bootstrap over names (cluster), averaging repeats within a name first.
        per_name = sub.groupby("name")["value"].mean()
        names = per_name.index.to_numpy()
        boot = np.array(
            [
                per_name.loc[rng.choice(names, size=len(names), replace=True)].mean()
                for _ in range(n_boot)
            ]
        )
        records.append(
            {
                "subgroup": subgroup,
                "political_mean": per_name.mean(),
                "political_lo": np.percentile(boot, 2.5),
                "political_hi": np.percentile(boot, 97.5),
                "n_names": len(names),
            }
        )
    return pd.DataFrame(records).set_index("subgroup")


def _plot_legibility(table: pd.DataFrame, out_path: Path) -> None:
    subgroups = [s for s in SUBGROUP_ORDER if s in table.index]
    x = np.arange(len(subgroups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, table.loc[subgroups, "race_recall"], width, label="race recall", color="#3b6")
    ax.bar(x + width / 2, table.loc[subgroups, "gender_recall"], width, label="gender recall", color="#39c")
    ax.plot(x - width / 2, table.loc[subgroups, "race_abstain"], "kx", label="race 'cannot tell'")
    ax.plot(x + width / 2, table.loc[subgroups, "gender_abstain"], "k+", label="gender 'cannot tell'")
    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Name-cue legibility: recall of intended race / gender")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_political(table: pd.DataFrame, out_path: Path) -> None:
    subgroups = [s for s in SUBGROUP_ORDER if s in table.index]
    y = np.arange(len(subgroups))
    means = table.loc[subgroups, "political_mean"].to_numpy()
    lo = table.loc[subgroups, "political_lo"].to_numpy()
    hi = table.loc[subgroups, "political_hi"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="o", color="#825", capsize=4)
    ax.axvline(0.0, color="grey", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(subgroups)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("inferred political leaning  (-1 conservative … +1 liberal)")
    ax.set_title("Inferred political leaning from name alone, by subgroup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="cue_probe.csv from stage 05")
    parser.add_argument("--out-dir", default="results/cue_probe")
    parser.add_argument("--figures-dir", default="figures/cue_probe")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.probe)

    race = _recall_table(frame, "race", "intended_race")
    gender = _recall_table(frame, "gender", "intended_gender")
    political = _political_summary(frame, args.bootstrap, args.seed)
    table = race.join(gender, how="outer").join(political, how="outer")
    table = table.reindex([s for s in SUBGROUP_ORDER if s in table.index])
    table.to_csv(out_dir / "legibility_by_subgroup.csv")

    _confusion(frame, "race", "intended_race").to_csv(out_dir / "race_confusion.csv")
    _confusion(frame, "gender", "intended_gender").to_csv(out_dir / "gender_confusion.csv")

    pol = frame[frame["attribute"] == "political"].copy()
    pol["value"] = pd.to_numeric(pol["parsed_value"], errors="coerce")
    pol.groupby(["subgroup", "name"])["value"].mean().reset_index().to_csv(
        out_dir / "political_by_name.csv", index=False
    )

    _plot_legibility(table, fig_dir / "name_cue_legibility.png")
    _plot_political(table, fig_dir / "inferred_political_by_subgroup.png")

    print("Legibility summary:\n", table.round(3).to_string())
    print(f"\nWrote tables to {out_dir} and figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
