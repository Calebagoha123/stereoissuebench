#!/usr/bin/env python3
"""Recompute per-cell baseline_liberal_score (and cue_effect) using only
non-neutral baseline rows. Produces a corrected CSV for the conditional-on-
stance diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", default="data/processed/evaluated_nonneutral_with_effects.csv")
    p.add_argument("--out", dest="dst", default="data/processed/evaluated_nonneutral_recomputed.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.src, low_memory=False)

    cell = ["template_id", "ces_variable"]

    base_mask = df["cue_condition"].eq("baseline")
    base_means = (
        df.loc[base_mask, cell + ["liberal_score"]]
        .groupby(cell, as_index=False)["liberal_score"]
        .mean()
        .rename(columns={"liberal_score": "_new_baseline"})
    )
    base_counts = (
        df.loc[base_mask, cell + ["liberal_score"]]
        .groupby(cell, as_index=False)["liberal_score"]
        .size()
        .rename(columns={"size": "_n_baseline"})
    )

    df = df.merge(base_means, on=cell, how="left")
    df = df.merge(base_counts, on=cell, how="left")

    cells_before = df[cell].drop_duplicates().shape[0]
    no_baseline = df["_new_baseline"].isna()
    n_dropped_rows = int(no_baseline.sum())
    n_dropped_cells = df.loc[no_baseline, cell].drop_duplicates().shape[0]
    df = df.loc[~no_baseline].copy()
    cells_after = df[cell].drop_duplicates().shape[0]

    df["baseline_liberal_score"] = df["_new_baseline"]
    cued = df["cue_condition"].ne("baseline")
    df.loc[cued, "cue_effect"] = df.loc[cued, "liberal_score"] - df.loc[cued, "baseline_liberal_score"]
    df.loc[~cued, "cue_effect"] = 0.0

    df = df.drop(columns=["_new_baseline", "_n_baseline"])

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dst, index=False)

    print(f"input rows : {len(df) + n_dropped_rows:,}")
    print(f"output rows: {len(df):,}  (dropped {n_dropped_rows:,} rows in {n_dropped_cells} cells with no surviving baseline)")
    print(f"cells: {cells_before} -> {cells_after}")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
