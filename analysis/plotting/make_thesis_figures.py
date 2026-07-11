#!/usr/bin/env python3
"""Three headline thesis figures for the cue-steering run (3 open-source models).

Classifier of record: DeBERTa ``bert_liberal_score`` in {-1, 0, +1}
(+1 = wrote the liberal side, -1 = conservative side, 0 = neutral). Titles are
omitted deliberately (a LaTeX caption carries the takeaway).

  fig1_forest.png        Shift in mean liberal score vs. the no-cue baseline for
                         every cue, grouped by cue family, one marker + 95% CI
                         per model. CIs are clustered on CES issue.
  fig2_calibration.png   Model stance shift (cued - baseline) vs. the REAL CES
                         2025 subgroup shift (subgroup - population). y = x is
                         perfect calibration; steeper than the line = exaggerates
                         the real gap, flatter (toward y = 0) = flattens it.
                         Colour = model, marker shape = cue family.
  fig3_composition.png   Conservative / Neutral / Liberal response mix per cue
                         and model, as stacked bars.

Reads results/full_3x/bert_eval_<model>.csv (slim per-model stance scores) and
results/full_3x/ces_estimates.csv (weighted CES 2025 estimates from the .dta).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 3 open-source models (GPT arm not yet available). Order = legend order.
MODELS = ["qwen", "gemma", "llama"]
MODEL_LABEL = {"qwen": "Qwen-3.6-27B", "gemma": "Gemma-3-12B", "llama": "Llama-3.1-8B"}
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#CC79A7"}  # Okabe-Ito

SCORE = "bert_liberal_score"
Z = 1.96

# Composition palette
C_CON, C_NEU, C_LIB = "#B2182B", "#DBDBDB", "#3A6EA5"


def _save(fig, out: Path, stem: str, fmts) -> None:
    """Write a figure in every requested format. PDF is vector (text stays
    selectable and scales without pixelation) and is the right choice for LaTeX;
    PNG is kept for quick previewing."""
    for ext in fmts:
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data + cue masks
# --------------------------------------------------------------------------- #
def load(results_dir: Path) -> dict[str, pd.DataFrame]:
    cols = ["arm", "cue_condition", "cue_family", "cue_group", "issue_id", SCORE]
    return {m: pd.read_csv(results_dir / f"bert_eval_{m}.csv", usecols=cols,
                           low_memory=False) for m in MODELS}


def base_mask(df: pd.DataFrame):
    return (df.arm == "A") & (df.cue_condition == "baseline")


def cue_mask(df: pd.DataFrame, family: str, group: str):
    if family.startswith("explicit"):
        return (df.arm == "A") & (df.cue_condition == f"{family}_{group}")
    # implicit cues live in arm B, identified by family + group (cue_condition
    # carries the per-instance name/state suffix, so we don't match on it).
    return (df.arm == "B") & (df.cue_family == family) & (df.cue_group == group)


# --- estimators: 95% CI clustered on issue -------------------------------- #
def _issue_means(df, mask):
    return df[mask].groupby("issue_id")[SCORE].mean()


def shift_ci(df, cued_mask, bmask):
    d = (_issue_means(df, cued_mask) - _issue_means(df, bmask)).dropna()
    pooled = df[cued_mask][SCORE].mean() - df[bmask][SCORE].mean()
    return pooled, Z * d.std(ddof=1) / np.sqrt(len(d))


# Cue rows for the forest plot: (band label, family, group, row label)
FOREST_BANDS = [
    ("EXPLICIT\nPOLITICAL", [
        ("explicit_political", "democrat", "Democrat"),
        ("explicit_political", "independent", "Independent"),
        ("explicit_political", "republican", "Republican")]),
    ("EXPLICIT\nDEMOGRAPHIC", [
        ("explicit_demographic", "black_woman", "Black woman"),
        ("explicit_demographic", "black_man", "Black man"),
        ("explicit_demographic", "white_woman", "White woman"),
        ("explicit_demographic", "white_man", "White man")]),
    ("IMPLICIT\nPOLITICAL", [
        ("implicit_political", "blue_state", "Blue state"),
        ("implicit_political", "swing_state", "Swing state"),
        ("implicit_political", "red_state", "Red state")]),
    ("IMPLICIT\nDEMOGRAPHIC", [
        ("implicit_demographic", "black_woman", "Name: Black woman"),
        ("implicit_demographic", "black_man", "Name: Black man"),
        ("implicit_demographic", "white_woman", "Name: White woman"),
        ("implicit_demographic", "white_man", "Name: White man")]),
]


# --------------------------------------------------------------------------- #
# Fig 1 — forest
# --------------------------------------------------------------------------- #
def fig_forest(data, out: Path, fmts):
    # Lay out rows top -> bottom with a gap between bands.
    rows, bands, y = [], [], 0.0
    for band, items in FOREST_BANDS:
        y0 = y
        for fam, grp, lbl in items:
            rows.append((y, fam, grp, lbl))
            y += 1.0
        bands.append((band, (y0 + y - 1.0) / 2.0, y0 - 0.5, y - 0.5))
        y += 0.9  # gap

    fig, ax = plt.subplots(figsize=(9.6, 9.2))
    offs = np.linspace(0.24, -0.24, len(MODELS))  # model jitter within a row

    # alternating band shading
    for i, (_, _, lo, hi) in enumerate(bands):
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#000000", alpha=0.035, zorder=0)

    for yy, fam, grp, _ in rows:
        for m, off in zip(MODELS, offs):
            df = data[m]
            sh, e = shift_ci(df, cue_mask(df, fam, grp), base_mask(df))
            ax.errorbar(sh, yy + off, xerr=e, fmt="o", ms=6.5,
                        color=MODEL_COLOUR[m], ecolor=MODEL_COLOUR[m],
                        elinewidth=1.3, capsize=0, zorder=3, mec="white", mew=0.7)

    ax.axvline(0, color="#333333", lw=1.1, zorder=2)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[3] for r in rows], fontsize=10)
    ax.set_ylim(max(r[0] for r in rows) + 0.7, min(r[0] for r in rows) - 0.7)
    ax.tick_params(axis="y", length=0)

    # band labels in the left margin
    for band, yc, _, _ in bands:
        ax.text(-0.30, yc, band, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, fontweight="bold",
                color="#555555")

    lim = max(abs(x) for r in rows for x in [
        shift_ci(data[m], cue_mask(data[m], r[1], r[2]), base_mask(data[m]))[0]
        for m in MODELS]) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"Shift in mean liberal score vs. no-cue baseline  ($\hat{\Delta}_k$)",
                  fontsize=11.5)
    ax.text(0.0, 1.01, "more conservative  ←", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.5, color="#888888")
    ax.text(1.0, 1.01, "→  more liberal", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color="#888888")

    handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOUR[m],
                          label=MODEL_LABEL[m], mec="white", mew=0.7) for m in MODELS]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.94, bottom=0.07)
    _save(fig, out, "fig1_forest", fmts)


# --------------------------------------------------------------------------- #
# Fig 2 — CES calibration scatter
# --------------------------------------------------------------------------- #
FAM_MARKER = {
    "explicit_political": "o",
    "explicit_demographic": "s",
    "implicit_political": "^",
    "implicit_demographic": "D",
}
FAM_MARKER_LABEL = {
    "explicit_political": "Explicit political",
    "explicit_demographic": "Explicit demographic",
    "implicit_political": "Implicit political",
    "implicit_demographic": "Implicit demographic",
}


def fig_calibration(data, ces_table: Path, out: Path, fmts):
    ces = pd.read_csv(ces_table)
    fig, ax = plt.subplots(figsize=(9.0, 8.6))

    xs_all, ys_all = [], []
    for _, r in ces.iterrows():
        fam, grp = r.cue_family, r.cue_group
        x = r.ces_shift_mean
        for m in MODELS:
            df = data[m]
            sh, _ = shift_ci(df, cue_mask(df, fam, grp), base_mask(df))
            ax.scatter(x, sh, marker=FAM_MARKER[fam], s=64,
                       color=MODEL_COLOUR[m], edgecolor="white", linewidth=0.6,
                       zorder=3, alpha=0.9)
            xs_all.append(x); ys_all.append(sh)

    lim = max(0.3, np.nanmax(np.abs(xs_all + ys_all)) * 1.15)
    # "exaggerates" wedges: steeper than y=x (between the line and the y-axis),
    # i.e. the model's subgroup gap is bigger than the real CES gap.
    ax.fill_between([-lim, 0], [-lim, 0], [-lim, -lim], color="#B2182B", alpha=0.05, zorder=0)
    ax.fill_between([0, lim], [lim, lim], [0, lim], color="#B2182B", alpha=0.05, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color="#444444", ls="--", lw=1.2, zorder=1)
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)

    ax.text(-0.52 * lim, -0.86 * lim, "EXAGGERATES", color="#B2182B",
            fontsize=10, fontweight="bold", ha="center", va="center")
    ax.text(0.62 * lim, 0.02 * lim, "FLATTENS", color="#666666", fontsize=10,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.60 * lim, 0.72 * lim, "perfect calibration\n(model shift = real shift)",
            color="#666666", fontsize=9, ha="center", va="center", rotation=45,
            rotation_mode="anchor")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Real opinion shift in CES 2025  ($\mu_k^{CES}-\mu_{pop}^{CES}$)",
                  fontsize=11.5)
    ax.set_ylabel(r"Model stance shift  ($\bar{Y}_k-\bar{Y}_{baseline}$)", fontsize=11.5)

    model_handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOUR[m],
                                label=MODEL_LABEL[m], mec="white", mew=0.5)
                     for m in MODELS]
    fam_handles = [plt.Line2D([], [], marker=FAM_MARKER[f], ls="", color="#555555",
                              label=FAM_MARKER_LABEL[f], mec="white", mew=0.5)
                   for f in FAM_MARKER]
    leg1 = ax.legend(handles=model_handles, loc="upper left", frameon=False, fontsize=9.5)
    ax.add_artist(leg1)
    ax.legend(handles=fam_handles, loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    _save(fig, out, "fig2_calibration", fmts)


# --------------------------------------------------------------------------- #
# Fig 3 — Conservative / Neutral / Liberal composition
# --------------------------------------------------------------------------- #
COMPOSITION_CUES = [
    ("baseline", "baseline", "No cue"),
    ("explicit_political", "democrat", "Democrat"),
    ("explicit_political", "independent", "Independent"),
    ("explicit_political", "republican", "Republican"),
    ("implicit_political", "blue_state", "Blue state"),
    ("implicit_political", "red_state", "Red state"),
]


def _composition(df, mask):
    s = df[mask][SCORE]
    n = len(s)
    if n == 0:
        return 0.0, 0.0, 0.0
    return (s == -1).sum() / n, (s == 0).sum() / n, (s == 1).sum() / n


def _pct_ints(fracs):
    """Round fractions to integer percents that sum to exactly 100
    (largest-remainder), so the three labels in a stack never read 99/101%."""
    raw = [f * 100 for f in fracs]
    out = [int(np.floor(x)) for x in raw]
    short = 100 - sum(out)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[:short]:
        out[i] += 1
    return out


def fig_composition(data, out: Path, fmts):
    cues = COMPOSITION_CUES
    fig, axes = plt.subplots(len(cues), len(MODELS),
                             figsize=(3.3 * len(MODELS), 0.92 * len(cues) + 1.2),
                             squeeze=False)
    for j, m in enumerate(MODELS):
        df = data[m]
        for i, (fam, grp, _) in enumerate(cues):
            ax = axes[i][j]
            mask = base_mask(df) if fam == "baseline" else cue_mask(df, fam, grp)
            con, neu, lib = _composition(df, mask)
            pcts = _pct_ints((con, neu, lib))
            left = 0.0
            for frac, pct, colour, txtcol in zip(
                    (con, neu, lib), pcts,
                    (C_CON, C_NEU, C_LIB), ("white", "#444444", "white")):
                ax.barh(0, frac, left=left, height=0.62, color=colour)
                if pct > 0:
                    cx = left + frac / 2
                    if frac >= 0.14:  # wide enough to hold the label inside
                        ax.text(cx, 0, f"{pct}%", ha="center", va="center",
                                fontsize=9, color=txtcol)
                    else:  # sliver: drop the label just below the bar
                        ax.text(cx, -0.46, f"{pct}%", ha="center", va="top",
                                fontsize=7.5, color=colour)
                left += frac
            ax.set_xlim(0, 1); ax.set_ylim(-0.9, 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=11, pad=8)
            if j == 0:
                ax.set_ylabel(cues[i][2], rotation=0, ha="right", va="center",
                              fontsize=10.5)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=C_CON, label="Conservative"),
               plt.Line2D([], [], marker="s", ls="", ms=11, color=C_NEU, label="Neutral"),
               plt.Line2D([], [], marker="s", ls="", ms=11, color=C_LIB, label="Liberal")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.10,
                        hspace=0.35, wspace=0.12)
    _save(fig, out, "fig3_composition", fmts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full_3x")
    ap.add_argument("--ces-table", default="results/full_3x/ces_estimates.csv")
    ap.add_argument("--figures-dir", default="figures/full_3x")
    ap.add_argument("--format", default="both", choices=["pdf", "png", "both"],
                    help="output format(s); pdf is vector, best for LaTeX (default both)")
    args = ap.parse_args()
    fmts = ["pdf", "png"] if args.format == "both" else [args.format]
    rd, fd = Path(args.results_dir), Path(args.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    data = load(rd)
    fig_forest(data, fd, fmts)
    fig_calibration(data, Path(args.ces_table), fd, fmts)
    fig_composition(data, fd, fmts)
    print(f"Wrote 3 figures to {fd} as {', '.join(fmts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
