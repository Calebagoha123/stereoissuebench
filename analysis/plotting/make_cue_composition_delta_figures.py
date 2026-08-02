#!/usr/bin/env python3
"""Cue-specific stance-composition figures.

For each cue, render the familiar issue x model composition view, but compare the
cue against its matched no-cue baseline:

  - faint upper bar = matched baseline composition
  - solid lower bar = cued composition
  - labels inside the solid segments = percentage-point change vs baseline for
    that stance class

So a '+14' label in the blue Liberal segment means the cue increased the share of
responses classified Liberal by 14 percentage points for that issue/model.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _common import EVAL_PREFIX, SCORE_COL  # classifier-of-record switch (SCORER env)

SCORE = SCORE_COL
MODELS = ["qwen", "gemma", "llama", "gpt56terra", "sonnet5"]
MODEL_LABEL = {
    "qwen": "Qwen-3.6-27B",
    "gemma": "Gemma-3-12B",
    "llama": "Llama-3.1-8B",
    "gpt56terra": "GPT-5.6 Terra",
    "sonnet5": "Claude Sonnet 5",
}

CUES = [
    ("explicit_political", "democrat", "Democrat"),
    ("explicit_political", "republican", "Republican"),
    ("explicit_political", "independent", "Independent"),
    ("explicit_demographic", "black_woman", "Black woman label"),
    ("explicit_demographic", "black_man", "Black man label"),
    ("explicit_demographic", "white_woman", "White woman label"),
    ("explicit_demographic", "white_man", "White man label"),
    ("implicit_political", "blue_state", "Blue state"),
    ("implicit_political", "swing_state", "Swing state"),
    ("implicit_political", "red_state", "Red state"),
    ("implicit_demographic", "black_woman", "Black-woman name"),
    ("implicit_demographic", "black_man", "Black-man name"),
    ("implicit_demographic", "white_woman", "White-woman name"),
    ("implicit_demographic", "white_man", "White-man name"),
]

# Exact stance palette from figures/full_3x/fig3_composition.
C_LIB = "#3A6EA5"
C_NEU = "#DBDBDB"
C_CON = "#B2182B"
TXT = "#222222"

# Okabe-Ito model palette, same assignment as make_thesis_figures.py, so a reader
# who has learned that Qwen is orange in Fig 1 reads these column headers without
# relearning anything.
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#0072B2",
                "gpt56terra": "#CC79A7", "sonnet5": "#56B4E9"}


def _darken(hex_colour: str, factor: float) -> tuple[float, float, float]:
    """Blend a hex colour toward black (factor 1.0 = unchanged), for using a marker
    palette as small text without losing which hue is which."""
    r, g, b = mcolors.to_rgb(hex_colour)
    return (r * factor, g * factor, b * factor)


def _thumb_marker():
    """Same vector thumbs-up marker used in make_thesis_figures.py."""
    v = np.array(
        [
            (0.02, 0.02),
            (0.02, 0.56),
            (0.12, 0.70),
            (0.12, 0.88),
            (0.19, 0.98),
            (0.30, 0.99),
            (0.37, 0.90),
            (0.35, 0.66),
            (0.33, 0.58),
            (0.82, 0.58),
            (0.95, 0.50),
            (0.95, 0.40),
            (0.88, 0.30),
            (0.93, 0.18),
            (0.84, 0.06),
            (0.86, 0.02),
            (0.30, 0.02),
            (0.02, 0.02),
        ],
        float,
    )
    v -= v.mean(0)
    v /= np.abs(v).max()
    return MplPath(v, closed=True)


THUMB = _thumb_marker()


def _theme() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 260,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )
    import sys as _s2, pathlib as _p2
    _s2.path.insert(0, str(_p2.Path(__file__).resolve().parent))
    import _style
    _style.apply(plt)  # Computer Modern, to match the thesis document


def _parse_prompt_id(pid: pd.Series) -> pd.DataFrame:
    parts = pid.str.split("__", expand=True)
    return pd.DataFrame({"template_id": parts[1], "rep": parts[3]}, index=pid.index)


def load_model(results_dir: Path, model: str) -> pd.DataFrame:
    cols = ["prompt_id", "cue_family", "cue_group", "issue_id", SCORE]
    df = pd.read_csv(results_dir / f"{EVAL_PREFIX}_{model}.csv", usecols=cols, low_memory=False)
    df = pd.concat([df, _parse_prompt_id(df["prompt_id"])], axis=1)
    df["model"] = model
    df["y"] = df[SCORE].astype(int)
    return df


def load_all(results_dir: Path) -> dict[str, pd.DataFrame]:
    return {model: load_model(results_dir, model) for model in MODELS}


def load_issue_labels(issues_csv: Path, present: set[str]) -> dict[str, str]:
    issues = pd.read_csv(issues_csv)
    labels = dict(zip(issues["ces_variable"], issues["ces_item_short"]))
    return {k: v for k, v in labels.items() if k in present}


def load_liberal_sign(issues_csv: Path) -> dict[str, int]:
    issues = pd.read_csv(issues_csv)
    return dict(zip(issues["ces_variable"], issues["liberal_sign"].astype(int)))


def class_shares(s: pd.Series) -> tuple[float, float, float]:
    n = len(s)
    if n == 0:
        return np.nan, np.nan, np.nan
    return (s.eq(1).mean(), s.eq(0).mean(), s.eq(-1).mean())


def matched_composition(df: pd.DataFrame, cue_family: str, cue_group: str) -> pd.DataFrame:
    keys = ["issue_id", "template_id", "rep"]
    base = df[df["cue_family"] == "baseline"][keys + ["y"]].rename(columns={"y": "base_y"})
    cue = df[(df["cue_family"] == cue_family) & (df["cue_group"] == cue_group)][keys + ["y"]].rename(
        columns={"y": "cue_y"}
    )
    paired = cue.merge(base, on=keys, how="inner")
    rows = []
    for issue, chunk in paired.groupby("issue_id", sort=False):
        base_lib, base_neu, base_con = class_shares(chunk["base_y"])
        cue_lib, cue_neu, cue_con = class_shares(chunk["cue_y"])
        rows.append(
            {
                "issue_id": issue,
                "n_pairs": len(chunk),
                "base_liberal": base_lib,
                "base_neutral": base_neu,
                "base_conservative": base_con,
                "cue_liberal": cue_lib,
                "cue_neutral": cue_neu,
                "cue_conservative": cue_con,
                "delta_liberal_pp": 100 * (cue_lib - base_lib),
                "delta_neutral_pp": 100 * (cue_neu - base_neu),
                "delta_conservative_pp": 100 * (cue_con - base_con),
            }
        )
    return pd.DataFrame(rows)


def baseline_issue_order(data: dict[str, pd.DataFrame], labels: dict[str, str]) -> list[str]:
    def comp(df: pd.DataFrame, issue: str) -> tuple[float, float, float]:
        s = df[(df["cue_family"] == "baseline") & (df["issue_id"] == issue)]["y"]
        return class_shares(s)

    return sorted(
        labels,
        key=lambda issue: np.mean([comp(data[m], issue)[0] - comp(data[m], issue)[2] for m in MODELS]),
        reverse=True,
    )


def _pct(frac: float) -> int:
    return int(round(100 * frac))


# Row geometry, named so the issue label and thumbs-up can be centred on the bars
# rather than on a hand-tuned offset. barh() takes y as the bar CENTRE, so the
# faint baseline bar spans [BASE_Y - BASE_H/2, BASE_Y + BASE_H/2] and the solid
# cued bar [CUE_Y - CUE_H/2, CUE_Y + CUE_H/2]; ROW_MID is the midpoint of the pair.
BASE_Y, BASE_H = 0.26, 0.22
CUE_Y, CUE_H = -0.12, 0.48
ROW_MID = ((BASE_Y + BASE_H / 2) + (CUE_Y - CUE_H / 2)) / 2
YLIM = (-0.48, 0.48)


# A segment narrower than this cannot hold "+14%" at the label size. Matches
# LABEL_MIN in fig3_composition; the three shares sum to 1, so at least one segment
# of every bar always clears it.
LABEL_MIN = 0.10


def draw_stack(ax, y: float, fracs: tuple[float, float, float], height: float, alpha: float, labels=None) -> None:
    left = 0.0
    colours = [C_LIB, C_NEU, C_CON]
    # Same three text colours as fig3_composition. No bold: the figures are set in
    # Computer Modern, which has no bold face (see _style), so weight is a no-op.
    txt_cols = ["white", "#4A4A4A", "white"]
    for idx, (frac, colour, txt_col) in enumerate(zip(fracs, colours, txt_cols)):
        ax.barh(y, frac, left=left, height=height, color=colour, alpha=alpha, edgecolor="none")
        if labels is not None:
            text = labels[idx]
            if text and frac >= LABEL_MIN:
                ax.text(
                    left + frac / 2,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    color=txt_col,
                )
        left += frac


def delta_label(delta_pp: float, threshold: float) -> str:
    """Change in percentage points for one stance class, as it appears inside the bar.

    Every segment wide enough to hold text gets a label, including the ones the cue
    did not move: with a suppression threshold, an issue where nothing happened drew
    a bar with no numbers at all, which reads as missing data rather than as no
    change. A move that rounds to nothing prints as a plain '0%', without a sign it
    has not earned."""
    if abs(delta_pp) < threshold:
        return ""
    rounded = round(delta_pp)
    return "0%" if rounded == 0 else f"{rounded:+.0f}%"


def plot_cue(
    comp_by_model: dict[str, pd.DataFrame],
    labels: dict[str, str],
    issues: list[str],
    cue_display: str,
    cue_family: str,
    cue_group: str,
    out_dir: Path,
    label_threshold: float,
    liberal_sign: dict[str, int],
) -> None:
    fig, axes = plt.subplots(
        len(issues),
        len(MODELS),
        figsize=(3.12 * len(MODELS), 0.42 * len(issues) + 2.0),
        squeeze=False,
    )
    # Wrapped like fig3_composition, which lets the label gutter be narrow enough
    # that the bars start at the same place in both figures.
    display_labels = {
        issue: textwrap.fill(label, width=26, break_long_words=False, break_on_hyphens=False)
        for issue, label in labels.items()
    }
    for j, model in enumerate(MODELS):
        table = comp_by_model[model].set_index("issue_id")
        for i, issue in enumerate(issues):
            ax = axes[i][j]
            rec = table.loc[issue]
            base = (rec.base_liberal, rec.base_neutral, rec.base_conservative)
            cue = (rec.cue_liberal, rec.cue_neutral, rec.cue_conservative)
            deltas = (
                rec.delta_liberal_pp,
                rec.delta_neutral_pp,
                rec.delta_conservative_pp,
            )
            # Faint matched baseline above, solid cued composition below.
            draw_stack(ax, BASE_Y, base, BASE_H, 0.32)
            draw_stack(
                ax,
                CUE_Y,
                cue,
                CUE_H,
                1.0,
                labels=[delta_label(d, label_threshold) for d in deltas],
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(*YLIM)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                # Header in the model's own colour over a hairline rule in the
                # undarkened hue, as in fig3_composition: the text is darkened ~28%
                # toward black because the Okabe-Ito orange and sky blue read at
                # marker size but not as 10.5pt type on white. Offsets are in points
                # rather than axes fractions because a row here is barely 0.3in tall,
                # so an axes-relative offset would shrink with the row.
                rule = mtransforms.offset_copy(ax.transAxes, fig=fig, y=6, units="points")
                text = mtransforms.offset_copy(ax.transAxes, fig=fig, y=9, units="points")
                ax.plot(
                    [0.0, 1.0],
                    [1.0, 1.0],
                    transform=rule,
                    color=MODEL_COLOUR[model],
                    lw=1.3,
                    clip_on=False,
                    zorder=1,
                )
                ax.text(
                    0.5,
                    1.0,
                    MODEL_LABEL[model],
                    transform=text,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    color=_darken(MODEL_COLOUR[model], 0.72),
                )
            if j == 0:
                # x in axes coords, y in DATA coords, so the label and thumb centre
                # on the bar pair (ROW_MID). They were previously placed at y =
                # -0.02 in axes coords, i.e. just below the axes floor, which put
                # them about half a row low.
                rowtr = ax.get_yaxis_transform()
                thumb_col = C_LIB if liberal_sign[issue] > 0 else C_CON
                ax.plot(
                    -0.035,
                    ROW_MID,
                    marker=THUMB,
                    markersize=8.5,
                    color=thumb_col,
                    ls="",
                    transform=rowtr,
                    clip_on=False,
                )
                ax.text(
                    -0.075,
                    ROW_MID,
                    display_labels[issue],
                    ha="right",
                    va="center",
                    multialignment="right",
                    fontsize=8.8,
                    linespacing=0.95,
                    transform=rowtr,
                    clip_on=False,
                    color=TXT,
                )

    # One row of handles at fig3_composition's sizes and colours, with the faint
    # upper bar the only entry this figure adds to that vocabulary.
    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=10, color=C_LIB, label="Liberal"),
        plt.Line2D([], [], marker="s", ls="", ms=10, color=C_NEU, label="Neutral"),
        plt.Line2D([], [], marker="s", ls="", ms=10, color=C_CON, label="Conservative"),
        plt.Line2D([], [], color="#777777", lw=5, alpha=0.32, label="matched baseline (faint upper bar)"),
        plt.Line2D([], [], marker=THUMB, ls="", ms=9, color=C_LIB, label="liberal side supports the issue"),
        plt.Line2D([], [], marker=THUMB, ls="", ms=9, color=C_CON, label="conservative side supports the issue"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.8,
        handletextpad=0.5,
        labelcolor="#333333",
    )
    fig.subplots_adjust(left=0.125, right=0.995, top=0.955, bottom=0.075, hspace=0.42, wspace=0.05)
    stem = f"composition_delta_{cue_family}_{cue_group}"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/full_3x")
    parser.add_argument("--issues-csv", default="data/input/issues_experiment.csv")
    parser.add_argument("--figures-dir", default="figures/cue_composition_delta")
    # 0 = label every segment wide enough to hold text, so no bar comes out blank.
    parser.add_argument("--label-threshold", type=float, default=0.0)
    args = parser.parse_args()

    _theme()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(results_dir)
    present = set.union(*(set(data[m][data[m]["cue_family"] == "baseline"].issue_id.unique()) for m in MODELS))
    labels = load_issue_labels(Path(args.issues_csv), present)
    liberal_sign = load_liberal_sign(Path(args.issues_csv))
    issues = baseline_issue_order(data, labels)

    all_rows = []
    for cue_family, cue_group, cue_display in CUES:
        comp_by_model = {
            model: matched_composition(data[model], cue_family, cue_group)
            for model in MODELS
        }
        for model, df in comp_by_model.items():
            d = df.copy()
            d.insert(0, "model", model)
            d.insert(1, "cue_family", cue_family)
            d.insert(2, "cue_group", cue_group)
            d.insert(3, "cue_display", cue_display)
            all_rows.append(d)
        plot_cue(
            comp_by_model,
            labels,
            issues,
            cue_display,
            cue_family,
            cue_group,
            out_dir,
            args.label_threshold,
            liberal_sign,
        )
    pd.concat(all_rows, ignore_index=True).to_csv(out_dir / "cue_composition_delta_long.csv", index=False)
    print(f"Wrote {len(CUES)} cue-composition delta figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
