#!/usr/bin/env python3
"""Split the LLM-judge (GPT-5.6 luna) corpus scores into per-model eval files.

The luna re-scoring (stance_model/judge_corpus.py) writes one combined file,
results/full_3x/luna_eval_all.csv, with a single continuous ``judge_score`` in
[0,100] (support for stance_target, same 0-100 codebook as DeBERTa's raw
``bert_pred_stance``). The downstream pipeline keys off per-model filenames, so
this script splits by generation_model into results/full_3x/luna_eval_<model>.csv
mirroring the bert_eval_<model>.csv schema.

We carry three model-response representations so the RQ2 sensitivity script can
compare DeBERTa vs discretized-luna vs continuous-luna in one pass:

  luna_pred_stance   raw judge_score in [0,100]   (analogue of bert_pred_stance)
  luna_liberal_cont  ((s-50)/50)*liberal_sign in [-1,+1]  (compression-preserving)
  luna_liberal_disc  {-1,0,+1}*liberal_sign, [40,60] neutral band (DeBERTa reband)
  luna_collapsed_stance  support/neutral/oppose (pre-sign), for neutral-fraction

The neutral band [40,60] matches stance_model/metrics.py (NEU_LO, NEU_HI), i.e.
the sensitivity script's band_h10_default, so luna_liberal_disc is apples-to-apples
with the DeBERTa headline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NEU_LO, NEU_HI = 40.0, 60.0  # == stance_model/metrics.py

# generation_model string (as it appears in luna_eval_all) -> short pipeline key.
# OS models arrive as full HuggingFace cache paths; frontier as API ids.
MODEL_KEY = {
    "google--gemma-3-12b-it": "gemma",
    "meta-llama--Llama-3.1-8B-Instruct": "llama",
    "Qwen--Qwen3.6-27B": "qwen",
    "claude-sonnet-5": "sonnet5",
    "gpt-5.6-terra": "gpt56terra",
}

CARRY = ["prompt_id", "arm", "cue_condition", "cue_family", "cue_group",
         "issue_id", "ces_variable", "stance_target", "liberal_sign",
         "generation_model", "generation_repeat", "instance_id", "finish_reason"]


def model_key(gen: str) -> str | float:
    for frag, key in MODEL_KEY.items():
        if frag in gen:
            return key
    return np.nan


def collapse(s: float) -> str:
    if s < NEU_LO:
        return "oppose"
    if s > NEU_HI:
        return "support"
    return "neutral"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/full_3x/luna_eval_all.csv")
    p.add_argument("--out-dir", default="results/full_3x")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inp, low_memory=False)
    n0 = len(df)
    n_null = df["judge_score"].isna().sum()
    df = df.dropna(subset=["judge_score"]).copy()
    df["model"] = df["generation_model"].map(model_key)
    n_unmapped = df["model"].isna().sum()
    if n_unmapped:
        bad = df.loc[df["model"].isna(), "generation_model"].unique()[:5]
        raise SystemExit(f"{n_unmapped} rows have unmapped generation_model, e.g. {list(bad)}")

    s = df["judge_score"].astype(float)
    sign = df["liberal_sign"].astype(int)
    df["luna_pred_stance"] = s
    df["luna_liberal_cont"] = ((s - 50.0) / 50.0) * sign
    df["luna_collapsed_stance"] = [collapse(v) for v in s]
    support = np.where(s > NEU_HI, 1, np.where(s < NEU_LO, -1, 0))
    df["luna_liberal_disc"] = support * sign

    keep = [c for c in CARRY if c in df.columns] + [
        "luna_pred_stance", "luna_liberal_cont",
        "luna_collapsed_stance", "luna_liberal_disc"]

    print(f"read {n0} rows; dropped {n_null} null judge_score; kept {len(df)}")
    for key, sub in df.groupby("model"):
        out = out_dir / f"luna_eval_{key}.csv"
        sub[keep].to_csv(out, index=False)
        vc = sub["luna_collapsed_stance"].value_counts()
        frac_neu = vc.get("neutral", 0) / len(sub)
        print(f"{key}: {len(sub)} rows -> {out}   neutral={frac_neu:.1%}  "
              f"mean|cont|={sub['luna_liberal_cont'].abs().mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
