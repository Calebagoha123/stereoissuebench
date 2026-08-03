#!/usr/bin/env python3
"""Probe accuracy per layer, true vs random labels: the Neplenbroek et al. idiom.

Replicates Figure 2 of Neplenbroek et al. (arXiv:2505.16467), which plots probe
accuracy against layer depth for a true-label probe and a random-label control
probe (Hewitt and Liang 2019). Their panel set is one model per panel for a single
attribute; ours is the same, for race x gender, with the *name* cue overlaid on the
explicit label so the two operationalisations of the same attribute can be read off
one panel.

Why this and not the share-of-layers bars: per-class recall degenerates when a
four-class probe collapses onto a single class (that class scores recall 1.0), which
happens at 22%/25%/9% of Llama/Gemma/Qwen layers and lands disproportionately on
white_woman. Balanced accuracy is immune -- a collapsed predictor scores exactly
chance -- and the control curve makes the floor visible rather than assumed.

Reads results/probe_internal/<tag>_decodability_by_layer.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parent))
import _style

_style.apply(plt)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

MODELS = ["gemma", "llama", "qwen"]
MODEL_LABEL = {"qwen": "Qwen-3.6-27B", "gemma": "Gemma-3-12B", "llama": "Llama-3.1-8B"}

TRUE_C = "#0072B2"   # Okabe-Ito blue, as the source figure
CTRL_C = "#E69F00"   # Okabe-Ito orange
NAME_C = "#009E73"   # Okabe-Ito green


def panel(ax, d, tag, show_ylabel):
    lab = d[(d.subset == "explicit_demographic") & (d.layer > 0)].sort_values("layer")
    nam = d[(d.subset == "name") & (d.layer > 0)].sort_values("layer")
    chance = float(lab.chance.iloc[0])

    ax.plot(lab.layer, lab.bal_acc, color=TRUE_C, lw=1.6,
            label="Explicit label, true")
    ax.plot(nam.layer, nam.bal_acc, color=NAME_C, lw=1.6, ls=(0, (4, 2)),
            label="Name, true")
    ax.plot(lab.layer, lab.control_acc, color=CTRL_C, lw=1.2,
            label="Random labels (control)")
    ax.axhline(chance, color="0.55", lw=0.8, ls=":", zorder=0)

    ax.set_title(MODEL_LABEL[tag], fontsize=11, pad=6)
    ax.set_xlabel("Layer")
    if show_ylabel:
        ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, lab.layer.max())
    ax.text(lab.layer.max() * 0.98, chance + 0.015, "chance", ha="right",
            va="bottom", fontsize=8, color="0.45")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/probe_internal")
    ap.add_argument("--out", default="figures/probe_thesis")
    a = ap.parse_args()

    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    for ax, tag in zip(axes, MODELS):
        panel(ax, pd.read_csv(res / f"{tag}_decodability_by_layer.csv"), tag,
              show_ylabel=(ax is axes[0]))

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10), fontsize=9.5)

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_p0_legibility_layers.{ext}", bbox_inches="tight")
    print(f"wrote {out}/fig_p0_legibility_layers.{{pdf,png}}")


if __name__ == "__main__":
    main()
