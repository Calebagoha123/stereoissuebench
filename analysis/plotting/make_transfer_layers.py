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

CEIL_C = "0.62"      # grey: reference, not a finding
XFER_C = "#0072B2"   # Okabe-Ito blue: the claim
CTRL_C = "#E69F00"   # Okabe-Ito orange: control


def panel(ax, tr, dec, tag, show_ylabel):
    tr = tr[tr.layer > 0].sort_values("layer")
    ceil = dec[(dec.subset == "explicit_demographic") & (dec.layer > 0)].sort_values("layer")
    chance = float(tr.chance.iloc[0])

    ax.plot(ceil.layer, ceil.bal_acc, color=CEIL_C, lw=1.1, ls=(0, (1, 1.6)),
            label="Explicit label, own task (ceiling)")
    ax.plot(tr.layer, tr.label_to_name, color=XFER_C, lw=1.8,
            label=r"Transfer: label $\rightarrow$ name")
    if "control" in tr.columns:
        ax.plot(tr.layer, tr.control, color=CTRL_C, lw=1.2,
                label="Shuffled labels (control)")
    ax.axhline(chance, color="0.55", lw=0.8, ls=":", zorder=0)

    ax.set_title(MODEL_LABEL[tag], fontsize=11, pad=6)
    ax.set_xlabel("Layer")
    if show_ylabel:
        ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, tr.layer.max())
    ax.text(tr.layer.max() * 0.98, chance + 0.015, "chance", ha="right",
            va="bottom", fontsize=8, color="0.45")
    m = tr.label_to_name.mean()
    ax.text(0.03, 0.90, f"mean {m:.2f}", transform=ax.transAxes, fontsize=9,
            color=XFER_C, ha="left", va="top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/probe_internal")
    ap.add_argument("--out", default="figures/probe_thesis")
    a = ap.parse_args()

    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    for ax, tag in zip(axes, MODELS):
        panel(ax,
              pd.read_csv(res / f"{tag}_cross_cue_transfer.csv"),
              pd.read_csv(res / f"{tag}_decodability_by_layer.csv"),
              tag, show_ylabel=(ax is axes[0]))

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10), fontsize=9.5)

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_p2_transfer_layers.{ext}", bbox_inches="tight")
    print(f"wrote {out}/fig_p2_transfer_layers.{{pdf,png}}")


if __name__ == "__main__":
    main()
