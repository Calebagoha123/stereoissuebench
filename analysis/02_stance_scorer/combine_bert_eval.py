#!/usr/bin/env python3
"""Concatenate arm-a + arm-b BERT stance scores per model into one slim file.

Reads bert_<model>_arm_{a,b}.csv (full predict.py output) from --in-dir and writes
results/full_3x/bert_eval_<model>.csv with only the columns the figures need, so
the file is small enough to sync down and plot locally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP = [
    "prompt_id", "arm", "cue_condition", "cue_family", "cue_group",
    "issue_id", "ces_variable", "stance_target", "liberal_sign",
    "generation_model", "finish_reason",
    "bert_pred_stance", "bert_collapsed_stance", "bert_support_score",
    "bert_liberal_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="/data/kell8360/full_3x_out")
    p.add_argument("--out-dir", default="results/full_3x")
    p.add_argument("--models", nargs="+", default=["llama", "gemma", "qwen"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir)
    for m in args.models:
        parts = []
        for arm in ("a", "b"):
            f = in_dir / f"bert_{m}_arm_{arm}.csv"
            df = pd.read_csv(f, low_memory=False)
            parts.append(df[[c for c in KEEP if c in df.columns]])
        combined = pd.concat(parts, ignore_index=True)
        out = out_dir / f"bert_eval_{m}.csv"
        combined.to_csv(out, index=False)
        print(f"{m}: {len(combined)} rows -> {out}")
        print(combined["bert_collapsed_stance"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
