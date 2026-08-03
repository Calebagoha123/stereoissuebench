#!/usr/bin/env python3
"""Cross-cue transfer per layer, in the Neplenbroek et al. (arXiv:2505.16467) idiom.

Their Figure 2 plots probe accuracy against layer with a random-label control
(Hewitt and Liang 2019). We plot the same axes for the quantity our RQ3 actually
rests on: a race x gender probe trained *only* on explicit-label prompts and applied
unchanged to name prompts, where no demographic word appears.

Three series per model:
  ceiling    the explicit-label probe on its own task -- what a working probe scores
  transfer   label -> name, the claim
  control    the same probe trained on shuffled labels, then transferred [PENDING]

Balanced accuracy throughout, not per-class recall: a four-class probe that collapses
onto a single class gives that class recall 1.0 and would be scored as a hit by any
per-class summary. That collapse occurs at 22%/25%/9% of Llama/Gemma/Qwen layers and
lands disproportionately on white_woman, which is what made the earlier
share-of-layers bars unusable. Balanced accuracy scores a collapsed predictor at
exactly chance.

Reads results/probe_internal/<tag>_{cross_cue_transfer,decodability_by_layer}.csv.
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

# Model colours as in make_thesis_figures.py / make_probe_thesis_figures.py, so a
# model keeps one identity across every figure in the thesis.
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#0072B2"}  # Okabe-Ito

# The two reference series stay neutral: they are the same quantity in all three
# panels, and any Okabe-Ito hue would collide with whichever model shares it.
CEIL_C = "0.72"      # ceiling: the probe on its own task
CTRL_C = "0.40"      # shuffled-label control


def panel(ax, tr, dec, ctrl, tag, show_ylabel):
    tr = tr[tr.layer > 0].sort_values("layer")
    ceil = dec[(dec.subset == "explicit_demographic") & (dec.layer > 0)].sort_values("layer")
    chance = float(tr.chance.iloc[0])

    colour = MODEL_COLOUR[tag]
    ax.plot(ceil.layer, ceil.bal_acc, color=CEIL_C, lw=1.3,
            label="Explicit label, own task (ceiling)")
    if ctrl is not None:
        c = ctrl[ctrl.layer > 0].sort_values("layer")
        ax.fill_between(c.layer, c.control_lo, c.control_hi, color=CTRL_C,
                        alpha=0.18, lw=0)
        ax.plot(c.layer, c.control_mean, color=CTRL_C, lw=1.1, ls=(0, (5, 2)),
                label="Shuffled labels (control)")
    ax.plot(tr.layer, tr.label_to_name, color=colour, lw=1.8,
            label=r"Transfer: label $\rightarrow$ name")
    ax.axhline(chance, color="0.55", lw=0.8, ls=":", zorder=0)

    ax.set_title(MODEL_LABEL[tag], fontsize=11, pad=6, color=colour)
    ax.set_xlabel("Layer")
    if show_ylabel:
        ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, tr.layer.max())
    ax.text(tr.layer.max() * 0.98, chance + 0.015, "chance", ha="right",
            va="bottom", fontsize=8, color="0.45")
    m = tr.label_to_name.mean()
    ax.text(0.03, 0.90, f"mean {m:.2f}", transform=ax.transAxes, fontsize=9,
            color=colour, ha="left", va="top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/probe_internal")
    ap.add_argument("--out", default="figures/probe_thesis")
    a = ap.parse_args()

    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    for ax, tag in zip(axes, MODELS):
        cpath = res / f"{tag}_transfer_control.csv"
        panel(ax,
              pd.read_csv(res / f"{tag}_cross_cue_transfer.csv"),
              pd.read_csv(res / f"{tag}_decodability_by_layer.csv"),
              pd.read_csv(cpath) if cpath.exists() else None,
              tag, show_ylabel=(ax is axes[0]))

    # The transfer line is model-coloured, so its legend key must not adopt any one
    # model's hue: draw the shared key as three short segments, one per model.
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    xfer_key = tuple(Line2D([], [], color=MODEL_COLOUR[t], lw=1.8) for t in MODELS)
    handles = [Line2D([], [], color=CEIL_C, lw=1.3),
               Line2D([], [], color=CTRL_C, lw=1.1, ls=(0, (5, 2))),
               xfer_key]
    labels = ["Explicit label, own task (ceiling)", "Shuffled labels (control)",
              r"Transfer: label $\rightarrow$ name"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10), fontsize=9.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_p2_transfer_layers.{ext}", bbox_inches="tight")
    print(f"wrote {out}/fig_p2_transfer_layers.{{pdf,png}}")


if __name__ == "__main__":
    main()
