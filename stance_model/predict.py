#!/usr/bin/env python3
"""Apply the trained stance cross-encoder to pipeline generations.

For each generated response we form a (proposition, response) pair and predict
stance on 0-100 (0 = opposes proposition, 100 = supports), then map onto the same
support / liberal scoring the Qwen judge produces, so this is a drop-in
alternative to pipeline/03_run_stance_eval.py.

Proposition string per row:
  - use a `proposition` column if the generations carry one, else build a
    declarative proposition from `stance_target` (the CES item's support target).
The proposition's support direction must match `topic_support`, so a high
predicted stance == support == support_score +1 (then *liberal_sign).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Same neutral band as evaluation; see stance_model/metrics.py.
NEU_LO, NEU_HI = 40.0, 60.0


def proposition_for(row: pd.Series) -> str:
    if isinstance(row.get("proposition"), str) and row["proposition"].strip():
        return row["proposition"].strip()
    return f"The government should support {row['stance_target']}."


def collapse(stance: float) -> str:
    if stance < NEU_LO:
        return "oppose"
    if stance > NEU_HI:
        return "support"
    return "neutral"


def support_score(stance: float) -> int:
    return {"support": 1, "neutral": 0, "oppose": -1}[collapse(stance)]


@torch.no_grad()
def predict(model, tokenizer, props, texts, device, batch_size, max_len) -> np.ndarray:
    preds = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            props[start : start + batch_size],
            texts[start : start + batch_size],
            truncation=True, max_length=max_len, padding=True, return_tensors="pt",
        ).to(device)
        out = model(**enc).logits.reshape(-1)
        preds.append(out.float().cpu().numpy())
    return np.clip(np.concatenate(preds), 0.0, 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Path to the saved final_model directory from train.py --mode final")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=384)
    args = parser.parse_args()

    path = Path(args.generations)
    df = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
    if "stance_target" not in df.columns and "proposition" not in df.columns:
        raise SystemExit("generations need a `stance_target` or `proposition` column")

    props = [proposition_for(r) for _, r in df.iterrows()]
    texts = df["response_text"].astype(str).tolist()

    device = args.device if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:  # noqa: BLE001 - DeBERTa-v3 slow-tokenizer fallback
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    stance = predict(model, tokenizer, props, texts, device, args.batch_size, args.max_len)
    df["bert_proposition"] = props
    df["bert_pred_stance"] = stance
    df["bert_collapsed_stance"] = [collapse(s) for s in stance]
    df["bert_support_score"] = [support_score(s) for s in stance]
    sign = df["liberal_sign"].astype(int) if "liberal_sign" in df.columns else 1
    df["bert_liberal_score"] = df["bert_support_score"] * sign

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Scored {len(df)} responses -> {out}")
    print(df["bert_collapsed_stance"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
