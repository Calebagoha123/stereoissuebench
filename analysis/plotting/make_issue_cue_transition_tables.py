#!/usr/bin/env python3
"""Issue x cue stance-class transition tables.

Each cell asks: among matched responses for this issue and cue, what is the most
common actual stance-class transition relative to the no-cue response?

Examples:
  N->L  = a response classified Neutral at baseline became Liberal under the cue
  L->C  = a response classified Liberal at baseline became Conservative

This is intentionally different from a mean-shift table. It shows changes in the
classified stance positions themselves.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
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
GREEN = "#009E73"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
GREY = "#F2F2F2"
TEXT = "#222222"

STATE = {-1: "C", 0: "N", 1: "L"}
TRANSITIONS = [
    (0, 1, "N->L", SKY, TEXT),
    (-1, 0, "C->N", GREEN, "white"),
    (-1, 1, "C->L", BLUE, "white"),
    (0, -1, "N->C", ORANGE, TEXT),
    (1, 0, "L->N", VERMILLION, "white"),
    (1, -1, "L->C", PURPLE, "white"),
]


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
    cols = ["prompt_id", "cue_family", "cue_group", "issue_id", "ces_variable", SCORE]
    df = pd.read_csv(results_dir / f"bert_eval_{model}.csv", usecols=cols, low_memory=False)
    df = pd.concat([df, parse_prompt_id(df["prompt_id"])], axis=1)
    df["model"] = model
    df["y"] = df[SCORE].astype(int)
    return df


def issue_order(results_dir: Path) -> pd.DataFrame:
    issues = pd.read_csv(results_dir / "ces_descriptives_issues.csv")
    return issues.sort_values("dem_rep_gap", ascending=False)[
        ["ces_variable", "issue", "dem_rep_gap"]
    ].reset_index(drop=True)


def compute_transitions(results_dir: Path) -> pd.DataFrame:
    rows = []
    issues = issue_order(results_dir)
    issue_label = issues.set_index("ces_variable")["issue"].to_dict()
    dem_rep_gap = issues.set_index("ces_variable")["dem_rep_gap"].to_dict()
    keys = ["ces_variable", "issue_id", "template_id", "rep"]

    for model in MODELS:
        df = load_model(results_dir, model)
        base = df[df["cue_family"] == "baseline"][keys + ["y"]].rename(columns={"y": "base_y"})
        for cue_family, cue_group, cue_display in CUES:
            cue = df[(df["cue_family"] == cue_family) & (df["cue_group"] == cue_group)][
                keys + ["y"]
            ].rename(columns={"y": "cue_y"})
            paired = cue.merge(base, on=keys, how="inner")
            for issue, chunk in paired.groupby("ces_variable", sort=False):
                n = len(chunk)
                unchanged = int((chunk["base_y"] == chunk["cue_y"]).sum())
                rec = {
                    "model": model,
                    "cue_family": cue_family,
                    "cue_group": cue_group,
                    "cue_display": cue_display.replace("\n", " "),
                    "ces_variable": issue,
                    "issue": issue_label.get(issue, issue),
                    "dem_rep_gap": dem_rep_gap.get(issue, np.nan),
                    "n_pairs": n,
                    "unchanged_pct": 100 * unchanged / n if n else np.nan,
                    "changed_pct": 100 * (n - unchanged) / n if n else np.nan,
                    "net_mean_shift": (chunk["cue_y"] - chunk["base_y"]).mean(),
                }
                for src, dst, label, _, _ in TRANSITIONS:
                    count = int(((chunk["base_y"] == src) & (chunk["cue_y"] == dst)).sum())
                    rec[label] = count
                    rec[f"{label}_pct"] = 100 * count / n if n else np.nan
                # Dominant nonzero transition, by share of all matched outputs.
                choices = [(label, rec[f"{label}_pct"]) for _, _, label, _, _ in TRANSITIONS]
                dominant_label, dominant_pct = max(choices, key=lambda x: x[1])
                rec["dominant_transition"] = dominant_label
                rec["dominant_transition_pct"] = dominant_pct
                rows.append(rec)
    return pd.DataFrame(rows)


def transition_style(label: str) -> tuple[str, str]:
    for _, _, lab, face, text in TRANSITIONS:
        if lab == label:
            return face, text
    return GREY, "#777777"


def plot_model_table(
    transitions: pd.DataFrame,
    issues: pd.DataFrame,
    model: str,
    out_dir: Path,
    min_pct: float,
) -> None:
    sub = transitions[transitions["model"] == model].copy()
    issue_labels = issues["issue"].tolist()
    n_rows = len(issue_labels)
    n_cols = len(CUES)

    lookup = sub.set_index(["issue", "cue_family", "cue_group"])
    fig = plt.figure(figsize=(15.6, 10.0))
    ax = fig.add_axes([0.23, 0.12, 0.75, 0.72])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, -1.85)
    ax.set_aspect("equal")
    ax.axis("off")

    for r, issue in enumerate(issue_labels):
        for c, (cue_family, cue_group, _) in enumerate(CUES):
            rec = lookup.loc[(issue, cue_family, cue_group)]
            label = rec["dominant_transition"]
            pct = float(rec["dominant_transition_pct"])
            if pct < min_pct:
                cell_text = "."
                face = GREY
                text_color = "#777777"
            else:
                cell_text = f"{label}\n{pct:.0f}%"
                face, text_color = transition_style(label)
            ax.add_patch(
                Rectangle((c, r), 1, 1, facecolor=face, edgecolor="white", linewidth=1.55)
            )
            ax.text(
                c + 0.5,
                r + 0.5,
                cell_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.2,
                fontweight="bold",
                linespacing=0.86,
            )

    for r, issue in enumerate(issue_labels):
        ax.text(-0.18, r + 0.5, issue, ha="right", va="center", fontsize=8.5, color=TEXT)

    for c, (_, _, display) in enumerate(CUES):
        ax.text(
            c + 0.5,
            -0.62,
            display,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=TEXT,
            fontweight="bold",
        )

    for label, start, end in GROUPS:
        width = end - start + 1
        ax.add_patch(
            Rectangle(
                (start, -1.35),
                width,
                0.38,
                facecolor="#EAEAEA",
                edgecolor="white",
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
            fontsize=8.1,
            color="#555555",
            clip_on=False,
        )
        if start > 0:
            ax.plot([start, start], [-1.35, n_rows], color="#D8D8D8", lw=1.0, clip_on=False)

    ax.text(-0.18, -0.62, "Issue", ha="right", va="bottom", fontsize=9, color="#555555", fontweight="bold")

    fig.suptitle(
        f"Issue x cue stance-class transition ledger: {MODEL_LABEL[model]}",
        fontsize=16,
        fontweight="bold",
        y=0.975,
        color=TEXT,
    )
    fig.text(
        0.61,
        0.940,
        (
            "Each cell shows the most common nonzero matched transition "
            f"if it occurs in at least {min_pct:.0f}% of responses for that issue/cue."
        ),
        ha="center",
        fontsize=9.6,
        color="#666666",
    )
    fig.text(
        0.23,
        0.905,
        "Rows include all 19 issues, sorted by CES Democrat-Republican gap.",
        ha="left",
        fontsize=8.6,
        color="#777777",
    )

    handles = [
        Patch(facecolor=SKY, edgecolor="white", label="N->L"),
        Patch(facecolor=GREEN, edgecolor="white", label="C->N"),
        Patch(facecolor=BLUE, edgecolor="white", label="C->L"),
        Patch(facecolor=ORANGE, edgecolor="white", label="N->C"),
        Patch(facecolor=VERMILLION, edgecolor="white", label="L->N"),
        Patch(facecolor=PURPLE, edgecolor="white", label="L->C"),
        Patch(facecolor=GREY, edgecolor="white", label=f". < {min_pct:.0f}%"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.61, 0.042),
        fontsize=8.7,
    )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"issue_cue_transition_ledger_{model}.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_wide_tables(transitions: pd.DataFrame, issues: pd.DataFrame, out_dir: Path) -> None:
    for model in MODELS:
        sub = transitions[transitions["model"] == model].copy()
        wide = pd.DataFrame({"issue": issues["issue"]})
        wide["ces_variable"] = issues["ces_variable"]
        wide["dem_rep_gap"] = issues["dem_rep_gap"]
        for cue_family, cue_group, display in CUES:
            vals = sub[(sub["cue_family"] == cue_family) & (sub["cue_group"] == cue_group)]
            mapping = {
                row.issue: f"{row.dominant_transition} {row.dominant_transition_pct:.1f}%"
                for row in vals.itertuples()
            }
            wide[display.replace("\n", " ")] = wide["issue"].map(mapping)
        wide.to_csv(out_dir / f"issue_cue_transition_ledger_{model}.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/full_3x")
    parser.add_argument("--figures-dir", default="figures/issue_cue_tables")
    parser.add_argument("--min-pct", type=float, default=8.0)
    args = parser.parse_args()

    _theme()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    issues = issue_order(results_dir)
    transitions = compute_transitions(results_dir)
    transitions.to_csv(out_dir / "issue_cue_transition_long.csv", index=False)
    write_wide_tables(transitions, issues, out_dir)
    for model in MODELS:
        plot_model_table(transitions, issues, model, out_dir, args.min_pct)
    print(f"Wrote issue x cue transition tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
