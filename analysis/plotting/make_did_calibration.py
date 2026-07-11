#!/usr/bin/env python3
"""DiD calibration figure: model stance shift (DeBERTa) vs real CES group shift.

Each point is a cue group. x = CES subgroup-minus-population liberal-score shift
(ground truth), y = model cued-minus-baseline shift (DeBERTa stance scorer,
pooled over the three generation models). The y=x line is perfect calibration:
- on the line  -> model reproduces the real group difference
- below (toward 0) -> model UNDER-personalises (compresses the real difference)
- above the line   -> model OVER-personalises (amplifies / stereotypes)
- wrong-sign quadrant -> model INVERTS the real difference

Reads results/full/rq2_bert_vs_ces.csv (built from bert_liberal_score + the
judge-independent CES subgroup scores). The Qwen judge is a placeholder; stance
of record is the DeBERTa cross-encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLORS = {
    "explicit_political": "#1F3A93",
    "explicit_demographic": "#C7372F",
    "implicit_political": "#58A9DE",
    "implicit_demographic": "#F0821E",
}
GROUP_LABELS = {
    "explicit_political": "Explicit political",
    "explicit_demographic": "Explicit demographic (label)",
    "implicit_political": "Implicit political (state)",
    "implicit_demographic": "Implicit demographic (name)",
}
# label text, plus a manual (dx, dy, ha) offset to declutter the origin cluster
POINT_LABELS = {
    ("explicit_political", "democrat"): ("Democrat", 0.014, -0.005, "left"),
    ("explicit_political", "republican"): ("Republican", 0.014, 0.0, "left"),
    ("explicit_political", "independent"): ("Independent", -0.014, -0.030, "right"),
    ("explicit_demographic", "black_woman"): ("Black woman", 0.016, 0.006, "left"),
    ("explicit_demographic", "black_man"): ("Black man", 0.016, 0.004, "left"),
    ("explicit_demographic", "white_woman"): ("White woman", 0.0, 0.028, "center"),
    ("explicit_demographic", "white_man"): ("White man", -0.016, -0.028, "right"),
    ("implicit_political", "blue_state"): ("Blue state", 0.0, 0.034, "center"),
    ("implicit_political", "red_state"): ("Red state", -0.016, 0.018, "right"),
    ("implicit_political", "swing_state"): ("Swing state", -0.016, -0.030, "right"),
    ("implicit_demographic", "black_woman"): ("“Black-female” name", 0.016, -0.018, "left"),
    ("implicit_demographic", "black_man"): ("“Black-male” name", 0.016, -0.024, "left"),
    ("implicit_demographic", "white_woman"): ("“White-female” name", 0.014, 0.020, "left"),
    ("implicit_demographic", "white_man"): ("“White-male” name", -0.016, 0.030, "right"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="results/full/rq2_bert_vs_ces.csv")
    p.add_argument("--out", default="figures/full/did_calibration.png")
    return p.parse_args()


def main() -> int:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    args = parse_args()
    df = pd.read_csv(args.table)
    df = df[df["cue_family"] != "baseline"].copy()

    fig, ax = plt.subplots(figsize=(9.2, 8.6))
    lim = 0.62
    # calibration line and reference axes
    ax.plot([-lim, lim], [-lim, lim], color="#444444", linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8, zorder=0)
    # shade the under-personalisation band (between line and y=0) lightly
    xs = np.linspace(-lim, lim, 200)
    ax.fill_between(xs, 0, xs, color="#000000", alpha=0.035, zorder=0)

    for _, r in df.iterrows():
        fam, grp = r["cue_family"], r["cue_group"]
        x, y = r["ces_shift_mean"], r["bert_shift"]
        ax.errorbar(
            x, y,
            yerr=[[y - r["bert_shift_lo"]], [r["bert_shift_hi"] - y]],
            xerr=[[x - r["ces_shift_ci_low"]], [r["ces_shift_ci_high"] - x]],
            fmt="o", markersize=10,
            color=GROUP_COLORS[fam], ecolor=GROUP_COLORS[fam],
            elinewidth=1.0, capsize=2.5, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3,
        )
        lab, dx, dy, ha = POINT_LABELS.get((fam, grp), (grp, 0.012, -0.022, "left"))
        ax.annotate(lab, (x, y), xytext=(x + dx, y + dy),
                    fontsize=8.5, ha=ha, color="#222222", zorder=4)

    ax.text(0.40, 0.40, "perfect\ncalibration (y = x)", rotation=45,
            fontsize=9, color="#444444", ha="center", va="center", rotation_mode="anchor")
    ax.text(0.46, 0.06, "under-\npersonalises", fontsize=9, color="#666666", ha="center")
    ax.text(0.06, 0.50, "over-\npersonalises", fontsize=9, color="#666666", ha="center")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Real CES group shift  (subgroup − population, liberal score)", fontsize=12)
    ax.set_ylabel("Model stance shift  (cued − baseline, DeBERTa liberal score)", fontsize=12)
    ax.set_title(
        "Does the model personalise toward real group differences?\n"
        "DiD calibration: model shift vs CES ground-truth shift.  95% bootstrap CIs.",
        loc="left", fontsize=14, pad=12,
    )
    handles = [mpatches.Patch(color=GROUP_COLORS[f], label=GROUP_LABELS[f]) for f in GROUP_LABELS]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
