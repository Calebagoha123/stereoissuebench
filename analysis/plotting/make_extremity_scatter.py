#!/usr/bin/env python3
"""Body figure: output-distribution compression as an extremity scatter.

One point per (model, cue-group) cell from composition_summary.csv: the mean
per-issue one-sidedness of the model's opinionated outputs (y) against the mean
per-issue one-sidedness of the real CES subgroup's opinions (x), extremity being
the distance of the liberal share from an even split (0 = evenly divided, 0.5 =
unanimous). Colour = model (Okabe-Ito), marker = cue family, with the no-cue
baseline drawn as a distinct open marker so the "compression predates the cues"
point is visible: baselines sit in the same high-extremity band as every cued
condition. Replaces the orphaned fig_within_group_flattening.pdf (no surviving
generator) with a reproducible one.

Reads results/robustness/composition_summary.csv.
Writes figures/robustness/extremity_scatter.{pdf,png}.
"""
from __future__ import annotations

import sys
import pathlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, MODEL_COLOUR, ROBUST  # noqa: E402

FAM_MARKER = {"explicit_political": "o", "explicit_demographic": "s",
              "implicit_political": "^", "implicit_demographic": "D"}
FAM_LABEL = {"explicit_political": "Party label",
             "explicit_demographic": "Race × gender label",
             "implicit_political": "State",
             "implicit_demographic": "Name"}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    s = pd.read_csv(ROBUST / "composition_summary.csv")
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    lim = 0.52
    ax.plot([0, lim], [0, lim], "--", color="#666", lw=1.2, zorder=1)
    ax.text(0.42, 0.435, "outputs as one-sided\nas the group divides", color="#777",
            fontsize=8.5, ha="center", va="bottom", rotation=38)

    for m in MODELS:
        d = s[(s.model == m) & (s.cue_family != "baseline")]
        for fam, g in d.groupby("cue_family"):
            ax.scatter(g["mean_ces_extremity"], g["mean_model_extremity"],
                       marker=FAM_MARKER[fam], s=46, color=MODEL_COLOUR[m],
                       alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
        b = s[(s.model == m) & (s.cue_family == "baseline")]
        # baseline vs the population split: open star, same colour
        ax.scatter(b["mean_ces_extremity"], b["mean_model_extremity"], marker="*",
                   s=230, facecolor="none", edgecolor=MODEL_COLOUR[m], linewidth=1.8,
                   zorder=4)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Real CES group one-sidedness per issue\n(distance of liberal share from 50/50, across respondents)",
                  fontsize=10.5)
    ax.set_ylabel("Model output one-sidedness per issue\n(distance of liberal share from 50/50, across prompts)",
                  fontsize=10.5)

    model_handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOUR[m],
                                label=MODEL_LABEL[m], mec="white", mew=0.5) for m in MODELS]
    fam_handles = [plt.Line2D([], [], marker=FAM_MARKER[f], ls="", color="#555",
                              label=FAM_LABEL[f], mec="white", mew=0.5) for f in FAM_MARKER]
    fam_handles.append(plt.Line2D([], [], marker="*", ls="", mfc="none", mec="#555",
                                  mew=1.6, ms=13, label="No-cue baseline"))
    leg1 = ax.legend(handles=model_handles, loc="upper left", frameon=False, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=fam_handles, loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path("figures/robustness/extremity_scatter")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Wrote {out}.pdf/.png ({len(s)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
