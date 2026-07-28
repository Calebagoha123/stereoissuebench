#!/usr/bin/env python3
"""Issue x cue stance-shift tables.

Each cell asks a simple question: for this issue, did this cue move the model's
stance relative to the matched no-cue baseline, and in which direction?

Cell values are matched issue-level mean shifts on the {-1, 0, +1} stance scale:

    mean(cued stance - baseline stance)

matched by issue, template id, and repeat id. Arm-B cues therefore use the same
35-template subset as their cue rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd


SCORE = "bert_liberal_score"
MODELS = ["qwen", "gemma", "llama", "gpt56terra", "sonnet5"]
MODEL_LABEL = {
    "qwen": "Qwen-3.6-27B",
    "gemma": "Gemma-3-12B",
    "llama": "Llama-3.1-8B",
    "gpt56terra": "GPT-5.6 Terra",
    "sonnet5": "Claude Sonnet 5",
}

CUES = [
    ("explicit_political", "democrat", "Dem"),
    ("explicit_political", "republican", "Rep"),
    ("explicit_political", "independent", "Ind"),
    ("explicit_demographic", "black_woman", "BW\nlabel"),
    ("explicit_demographic", "black_man", "BM\nlabel"),
    ("explicit_demographic", "white_woman", "WW\nlabel"),
    ("explicit_demographic", "white_man", "WM\nlabel"),
    ("implicit_political", "blue_state", "Blue\nstate"),
    ("implicit_political", "swing_state", "Swing\nstate"),
    ("implicit_political", "red_state", "Red\nstate"),
    ("implicit_demographic", "black_woman", "BW\nname"),
    ("implicit_demographic", "black_man", "BM\nname"),
    ("implicit_demographic", "white_woman", "WW\nname"),
    ("implicit_demographic", "white_man", "WM\nname"),
]

GROUPS = [
    ("party label", 0, 2),
    ("race x gender label", 3, 6),
    ("state", 7, 9),
    ("name", 10, 13),
]

# Okabe-Ito colorblind-safe palette.
BLUE = "#0072B2"
SKY = "#56B4E9"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
GREY = "#F2F2F2"
GRID = "#FFFFFF"
TEXT = "#222222"


def _theme() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 260,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def parse_prompt_id(pid: pd.Series) -> pd.DataFrame:
    parts = pid.str.split("__", expand=True)
    return pd.DataFrame({"template_id": parts[1], "rep": parts[3]}, index=pid.index)


def load_model(results_dir: Path, model: str) -> pd.DataFrame:
    cols = [
        "prompt_id",
        "arm",
        "cue_family",
        "cue_group",
        "issue_id",
        "ces_variable",
        SCORE,
    ]
    df = pd.read_csv(results_dir / f"bert_eval_{model}.csv", usecols=cols, low_memory=False)
    df = pd.concat([df, parse_prompt_id(df["prompt_id"])], axis=1)
    df["model"] = model
    df["y"] = df[SCORE].astype(float)
    return df


def issue_order(results_dir: Path) -> pd.DataFrame:
    issues = pd.read_csv(results_dir / "ces_descriptives_issues.csv")
    issues = issues.sort_values("dem_rep_gap", ascending=False)
    return issues[["ces_variable", "issue", "dem_rep_gap"]].reset_index(drop=True)


def compute_shifts(results_dir: Path) -> pd.DataFrame:
    rows = []
    issues = issue_order(results_dir)
    issue_label = issues.set_index("ces_variable")["issue"].to_dict()
    dem_rep_gap = issues.set_index("ces_variable")["dem_rep_gap"].to_dict()
    keys = ["ces_variable", "issue_id", "template_id", "rep"]
    for model in MODELS:
        df = load_model(results_dir, model)
        base = df[df["cue_family"] == "baseline"][keys + ["y"]].rename(columns={"y": "baseline_mean"})
        for family, group, display in CUES:
            cue = df[(df["cue_family"] == family) & (df["cue_group"] == group)][keys + ["y"]].rename(
                columns={"y": "cue_mean"}
            )
            paired = cue.merge(base, on=keys, how="inner")
            paired["delta"] = paired["cue_mean"] - paired["baseline_mean"]
            by_issue = (
                paired.groupby("ces_variable", as_index=False)
                .agg(
                    issue_shift=("delta", "mean"),
                    n_pairs=("delta", "size"),
                    baseline_mean=("baseline_mean", "mean"),
                    cue_mean=("cue_mean", "mean"),
                )
            )
            for rec in by_issue.to_dict("records"):
                rows.append(
                    {
                        "model": model,
                        "cue_family": family,
                        "cue_group": group,
                        "cue_display": display.replace("\n", " "),
                        "ces_variable": rec["ces_variable"],
                        "issue": issue_label.get(rec["ces_variable"], rec["ces_variable"]),
                        "dem_rep_gap": dem_rep_gap.get(rec["ces_variable"], np.nan),
                        "issue_shift": rec["issue_shift"],
                        "baseline_mean": rec["baseline_mean"],
                        "cue_mean": rec["cue_mean"],
                        "n_pairs": rec["n_pairs"],
                    }
                )
    return pd.DataFrame(rows)


def bucket(delta: float, small: float, large: float) -> tuple[str, str, str]:
    """Return label, facecolor, textcolor."""
    if delta >= large:
        return "L++", BLUE, "white"
    if delta >= small:
        return "L+", SKY, TEXT
    if delta <= -large:
        return "C++", VERMILLION, "white"
    if delta <= -small:
        return "C+", ORANGE, TEXT
    return ".", GREY, "#777777"


def plot_model_table(
    shifts: pd.DataFrame,
    issues: pd.DataFrame,
    model: str,
    out_dir: Path,
    small: float,
    large: float,
) -> None:
    sub = shifts[shifts["model"] == model].copy()
    issue_labels = issues["issue"].tolist()
    n_rows = len(issue_labels)
    n_cols = len(CUES)

    pivot = sub.pivot_table(
        index="issue",
        columns=["cue_family", "cue_group"],
        values="issue_shift",
        aggfunc="mean",
    )

    fig = plt.figure(figsize=(14.2, 9.8))
    ax = fig.add_axes([0.24, 0.13, 0.73, 0.70])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, -1.85)
    ax.set_aspect("equal")
    ax.axis("off")

    for r, issue in enumerate(issue_labels):
        for c, (family, group, _) in enumerate(CUES):
            delta = float(pivot.loc[issue, (family, group)])
            label, face, text_color = bucket(delta, small, large)
            ax.add_patch(
                Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=face,
                    edgecolor=GRID,
                    linewidth=1.6,
                )
            )
            ax.text(
                c + 0.5,
                r + 0.5,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    # Row labels.
    for r, issue in enumerate(issue_labels):
        ax.text(
            -0.18,
            r + 0.5,
            issue,
            ha="right",
            va="center",
            fontsize=8.6,
            color=TEXT,
        )

    # Cue labels.
    for c, (_, _, display) in enumerate(CUES):
        ax.text(
            c + 0.5,
            -0.62,
            display,
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=TEXT,
            fontweight="bold",
        )

    # Cue-family header bands.
    for label, start, end in GROUPS:
        width = end - start + 1
        ax.add_patch(
            Rectangle(
                (start, -1.35),
                width,
                0.38,
                facecolor="#EAEAEA",
                edgecolor=GRID,
                linewidth=1.2,
                clip_on=False,
            )
        )
        ax.text(
            start + width / 2,
            -1.16,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#555555",
            clip_on=False,
        )
        if start > 0:
            ax.plot([start, start], [-1.35, n_rows], color="#D8D8D8", lw=1.0, clip_on=False)

    # Left header.
    ax.text(
        -0.18,
        -0.62,
        "Issue",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
        fontweight="bold",
    )

    fig.suptitle(
        f"Issue x cue stance-shift ledger: {MODEL_LABEL[model]}",
        fontsize=16,
        fontweight="bold",
        y=0.975,
        color=TEXT,
    )
    fig.text(
        0.60,
        0.942,
        (
            "Cells show matched issue-level shift vs no-cue baseline. "
            f"L/C = more liberal/conservative; + if |shift| >= {small:.2f}, ++ if >= {large:.2f}."
        ),
        ha="center",
        fontsize=9.5,
        color="#666666",
    )
    fig.text(
        0.24,
        0.905,
        "Issues sorted by CES Democrat-Republican gap, largest first.",
        ha="left",
        fontsize=8.6,
        color="#777777",
    )

    legend_handles = [
        Patch(facecolor=BLUE, edgecolor="white", label=f"L++  >= +{large:.2f}"),
        Patch(facecolor=SKY, edgecolor="white", label=f"L+  +{small:.2f} to +{large:.2f}"),
        Patch(facecolor=GREY, edgecolor="white", label=f".  |shift| < {small:.2f}"),
        Patch(facecolor=ORANGE, edgecolor="white", label=f"C+  -{large:.2f} to -{small:.2f}"),
        Patch(facecolor=VERMILLION, edgecolor="white", label=f"C++  <= -{large:.2f}"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.60, 0.045),
        fontsize=8.8,
    )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"issue_cue_shift_ledger_{model}.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_wide_tables(shifts: pd.DataFrame, issues: pd.DataFrame, out_dir: Path) -> None:
    cue_cols = [(family, group, display.replace("\n", " ")) for family, group, display in CUES]
    for model in MODELS:
        sub = shifts[shifts["model"] == model]
        wide = pd.DataFrame({"issue": issues["issue"]})
        wide["ces_variable"] = issues["ces_variable"]
        wide["dem_rep_gap"] = issues["dem_rep_gap"]
        for family, group, display in cue_cols:
            vals = sub[(sub["cue_family"] == family) & (sub["cue_group"] == group)]
            mapping = vals.set_index("issue")["issue_shift"].to_dict()
            wide[display] = wide["issue"].map(mapping)
        wide.to_csv(out_dir / f"issue_cue_shift_ledger_{model}.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/full_3x")
    parser.add_argument("--figures-dir", default="figures/issue_cue_tables")
    parser.add_argument("--small-threshold", type=float, default=0.08)
    parser.add_argument("--large-threshold", type=float, default=0.20)
    args = parser.parse_args()

    _theme()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    issues = issue_order(results_dir)
    shifts = compute_shifts(results_dir)
    shifts.to_csv(out_dir / "issue_cue_shift_long.csv", index=False)
    write_wide_tables(shifts, issues, out_dir)
    for model in MODELS:
        plot_model_table(shifts, issues, model, out_dir, args.small_threshold, args.large_threshold)
    print(f"Wrote issue x cue shift tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
