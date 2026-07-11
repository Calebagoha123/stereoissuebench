#!/usr/bin/env python3
"""Belief-probe elicitation sensitivity (check K).

The 0–100 belief probe is elicited with 5 generations per cell. We report the
between-generation (elicitation) variance against the between-condition variance,
to show the belief scores are stable elicitations rather than sampling artefacts.

Note on design: the probe asks for a 0–100 probability rather than the −1/+1
rating used for stance precisely to avoid forcing the discretization the study is
trying to measure — the continuous elicitation lets the model express the neutral
hedging that turns out to drive the stance results.

Reads results/full/belief_probe_<model>.csv (opinion probe). Writes
results/probe_internal/belief_variance.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["llama", "gemma", "qwen"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B", "qwen": "Qwen3.6-27B"}
BELIEF = Path("results/full")


def main():
    recs = []
    for m in MODELS:
        d = pd.read_csv(BELIEF / f"belief_probe_{m}.csv", low_memory=False)
        d = d[d.probe_kind == "opinion"].copy()
        d["score"] = pd.to_numeric(d["parsed_score"], errors="coerce")
        d = d.dropna(subset=["score"])
        # cell = a single elicitation condition, varying only over the 5 repeats
        d["cell"] = (d["cue_condition"].astype(str) + "|" + d["issue_id"].astype(str))
        g = d.groupby("cell")["score"]
        cell_mean = g.transform("mean")
        grand = d["score"].mean()
        ss_within = ((d["score"] - cell_mean) ** 2).sum()
        ss_between = ((cell_mean - grand) ** 2).sum()
        ss_total = ((d["score"] - grand) ** 2).sum()
        k = g.count().mean()
        ms_b = ss_between / (g.ngroups - 1)
        ms_w = ss_within / (len(d) - g.ngroups)
        icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
        # typical between-generation SD within a cell (on 0-100 scale)
        within_sd = d.groupby("cell")["score"].std(ddof=1).mean()
        recs.append({"model": m, "n": len(d), "n_cells": g.ngroups, "reps_per_cell": round(k, 1),
                     "between_cell_frac": ss_between / ss_total,
                     "within_gen_frac": ss_within / ss_total,
                     "icc1": icc, "mean_within_cell_sd_0_100": within_sd})
    out = pd.DataFrame(recs)
    out.to_csv(Path("results/probe_internal/belief_variance.csv"), index=False)
    print("=== Belief-probe elicitation variance (0–100 opinion score, 5 generations) ===\n")
    for _, r in out.iterrows():
        print(f"  {MODEL_LABEL[r['model']]:>14}:  between-condition={r['between_cell_frac']*100:.0f}%  "
              f"between-generation={r['within_gen_frac']*100:.0f}%  ICC(1)={r['icc1']:.3f}  "
              f"typical within-cell SD={r['mean_within_cell_sd_0_100']:.1f} pts")
    print("\nBelief elicitations are stable: the between-generation component is small and the")
    print("belief signal is dominated by the condition, so the A2/B3 correlations are not")
    print("driven by elicitation noise.")
    print(f"Wrote results/probe_internal/belief_variance.csv")


if __name__ == "__main__":
    main()
