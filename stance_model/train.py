#!/usr/bin/env python3
"""Fine-tune a DeBERTa-v3 cross-encoder to regress proposition stance (0-100).

Input is a (proposition, paragraph) pair; target is writer_stance/100. Stance is
proposition-relative, so the proposition MUST be encoded alongside the text --
that is also what makes the model transferable to new topics.

Modes:
  --mode cv      Leave-propositions-out GroupKFold. This is the honest estimate
                 of how the model generalises to UNSEEN topics, i.e. the analogue
                 of applying it to the CES issues it never trained on. Reports
                 pooled out-of-fold metrics, overall and per paragraph_type.
  --mode final   Train on all data, save a single checkpoint for inference
                 (stance_model/predict.py).

Run on a GPU (Brains). Needs: torch, transformers>=4.40, sentencepiece, pandas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from metrics import stance_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "stance_model" / "dataset.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "stance_model"


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, props, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(
            list(props),
            list(texts),
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        # target scaled to [0,1] for stable regression; undone at eval time.
        self.labels = [float(x) / 100.0 for x in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.float)
        return item


def make_model(model_name: str):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )


def training_args(out_dir: str, args: argparse.Namespace, evaluate: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch" if evaluate else "no",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )


def predict_stance(trainer: Trainer, dataset: PairDataset) -> np.ndarray:
    raw = trainer.predict(dataset).predictions.reshape(-1)
    return np.clip(raw, 0.0, 1.0) * 100.0  # back to 0-100


def run_cv(df: pd.DataFrame, tokenizer, args: argparse.Namespace) -> None:
    groups = df["proposition_id"].to_numpy()
    gkf = GroupKFold(n_splits=args.folds)
    oof_pred = np.full(len(df), np.nan)

    for fold, (tr, va) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n=== Fold {fold + 1}/{args.folds}: "
              f"{df.iloc[tr]['proposition_id'].nunique()} train props, "
              f"{df.iloc[va]['proposition_id'].nunique()} val props ===")
        tr_ds = PairDataset(df.iloc[tr]["proposition"], df.iloc[tr]["text"],
                            df.iloc[tr]["writer_stance"], tokenizer, args.max_len)
        va_ds = PairDataset(df.iloc[va]["proposition"], df.iloc[va]["text"],
                            df.iloc[va]["writer_stance"], tokenizer, args.max_len)
        model = make_model(args.model)
        trainer = Trainer(
            model=model,
            args=training_args(f"{args.out}/cv_fold{fold}", args, evaluate=False),
            train_dataset=tr_ds,
            data_collator=lambda f: tokenizer.pad(f, return_tensors="pt"),
        )
        trainer.train()
        oof_pred[va] = predict_stance(trainer, va_ds)
        del trainer, model
        torch.cuda.empty_cache()

    df = df.assign(pred_stance=oof_pred)
    out_csv = Path(args.out) / "cv_oof_predictions.csv"
    df.to_csv(out_csv, index=False)

    gold = df["writer_stance"].to_numpy()
    report = {"overall": stance_metrics(oof_pred, gold)}
    for ptype, sub in df.groupby("paragraph_type"):
        report[f"paragraph_type={ptype}"] = stance_metrics(
            sub["pred_stance"].to_numpy(), sub["writer_stance"].to_numpy()
        )
    out_json = Path(args.out) / "cv_metrics.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("\n===== Leave-propositions-out (cross-topic) metrics =====")
    for k, v in report.items():
        print(f"\n[{k}]")
        for mk, mv in v.items():
            print(f"  {mk:14s} {mv:.4f}" if isinstance(mv, float) else f"  {mk:14s} {mv}")
    print(f"\nWrote {out_csv}\nWrote {out_json}")


def run_final(df: pd.DataFrame, tokenizer, args: argparse.Namespace) -> None:
    ds = PairDataset(df["proposition"], df["text"], df["writer_stance"],
                     tokenizer, args.max_len)
    model = make_model(args.model)
    trainer = Trainer(
        model=model,
        args=training_args(f"{args.out}/final", args, evaluate=False),
        train_dataset=ds,
        data_collator=lambda f: tokenizer.pad(f, return_tensors="pt"),
    )
    trainer.train()
    save_dir = Path(args.out) / "final_model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"\nSaved final model to {save_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cv", "final"], default="cv")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["proposition", "text", "writer_stance"]).reset_index(drop=True)
    print(f"Loaded {len(df)} examples / {df['proposition_id'].nunique()} propositions")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.mode == "cv":
        run_cv(df, tokenizer, args)
    else:
        run_final(df, tokenizer, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
