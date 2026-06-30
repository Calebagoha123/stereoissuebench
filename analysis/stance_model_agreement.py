#!/usr/bin/env python3
"""Agreement between the trained stance cross-encoder and the Qwen judge labels.

We run the cross-encoder (stance_model/predict.py) over the SAME model-generated
responses the Qwen judge already labelled, then measure how well the two agree.
This is a validation of the cheap local classifier against the existing judge —
NOT a re-run of the judge.

Inputs: results/full/bert_eval_{model}.csv, each carrying both
  - Qwen judge:   collapsed_stance, support_score, eval_label, liberal_score
  - cross-encoder: bert_collapsed_stance, bert_support_score, bert_pred_stance,
                   bert_liberal_score
Per generation model (llama/gemma/qwen) and pooled, reports agreement on the
3-way stance, the support score, and the signed liberal score (the quantity the
thesis actually uses), plus a confusion matrix and the refusal slice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

MODELS = ["llama", "gemma", "qwen"]
STANCE_ORDER = ["support", "neutral", "oppose"]  # ordinal for weighted kappa
JUDGE_NONLABEL = {"refusal", "PARSE_ERROR"}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def agree_block(df: pd.DataFrame) -> dict:
    """Agreement on rows where the judge gave a real 3-way label."""
    judged = df[~df["collapsed_stance"].isin(JUDGE_NONLABEL)].copy()
    judged = judged.dropna(subset=["collapsed_stance", "bert_collapsed_stance"])
    n = len(judged)
    out = {"n_compared": n, "n_total": len(df)}
    if n == 0:
        return out

    q3 = judged["collapsed_stance"].map(STANCE_ORDER.index).to_numpy()
    b3 = judged["bert_collapsed_stance"].map(STANCE_ORDER.index).to_numpy()
    out["stance_acc"] = float((q3 == b3).mean())
    out["stance_kappa"] = cohen_kappa_score(q3, b3)
    out["stance_kappa_w"] = cohen_kappa_score(q3, b3, weights="quadratic")

    # support score (-1/0/1) — judge may be NaN where collapsed was dropped already
    ss = judged.dropna(subset=["support_score"])
    out["support_acc"] = float(
        (ss["support_score"].astype(int) == ss["bert_support_score"].astype(int)).mean()
    )
    # signed liberal score (the downstream quantity)
    ls = judged.dropna(subset=["liberal_score"])
    out["liberal_acc"] = float(
        (ls["liberal_score"].astype(int) == ls["bert_liberal_score"].astype(int)).mean()
    )
    # continuous BERT stance vs judge support score (+1 support ↔ high stance)
    out["spearman_stance_vs_support"] = _spearman(
        judged["bert_pred_stance"].to_numpy(),
        judged["support_score"].astype(float).to_numpy(),
    )
    out["cm"] = confusion_matrix(q3, b3, labels=[0, 1, 2])  # rows=judge, cols=bert
    return out


def refusal_block(df: pd.DataFrame) -> pd.Series | None:
    ref = df[df["collapsed_stance"] == "refusal"]
    if ref.empty:
        return None
    return ref["bert_collapsed_stance"].value_counts(normalize=True)


def fmt(d: dict) -> str:
    if "stance_acc" not in d:
        return f"n={d['n_compared']}/{d['n_total']}  (no judged rows)"
    return (
        f"n={d['n_compared']:>6}/{d['n_total']:<6}  "
        f"stance: acc={d['stance_acc']:.3f} κ={d['stance_kappa']:.3f} κw={d['stance_kappa_w']:.3f}  "
        f"support_acc={d['support_acc']:.3f}  liberal_acc={d['liberal_acc']:.3f}  "
        f"ρ(stance,support)={d['spearman_stance_vs_support']:.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full")
    ap.add_argument("--out", default="docs/stance_model_agreement.md")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    frames = {}
    for m in MODELS:
        p = rdir / f"bert_eval_{m}.csv"
        if p.exists():
            frames[m] = pd.read_csv(p)
        else:
            print(f"[skip] missing {p}")
    if not frames:
        raise SystemExit("no bert_eval_*.csv found — run predict.py first")

    lines = ["# Cross-encoder vs Qwen judge — agreement\n",
             "Stance: support/neutral/oppose (judge refusals/parse-errors excluded). "
             "`liberal_acc` is exact agreement on the signed liberal score (the thesis quantity).\n"]
    pooled = pd.concat(frames.values(), ignore_index=True)
    print("=" * 100)
    for m, df in frames.items():
        d = agree_block(df)
        print(f"{m:<8} {fmt(d)}")
        lines.append(f"- **{m}** — {fmt(d)}")
        ref = refusal_block(df)
        if ref is not None:
            rs = ", ".join(f"{k}:{v:.0%}" for k, v in ref.items())
            print(f"         judge-refusal rows (n={int((df['collapsed_stance']=='refusal').sum())}) "
                  f"→ BERT says: {rs}")
    dp = agree_block(pooled)
    print("-" * 100)
    print(f"{'POOLED':<8} {fmt(dp)}")
    lines.append(f"- **pooled** — {fmt(dp)}")

    print("\nPooled confusion (rows=Qwen judge, cols=BERT), order=[support,neutral,oppose]:")
    cm = dp["cm"]
    hdr = "            " + "".join(f"{s:>9}" for s in STANCE_ORDER)
    print(hdr)
    for i, s in enumerate(STANCE_ORDER):
        print(f"  {s:>8} " + "".join(f"{cm[i, j]:>9}" for j in range(3)))

    lines.append("\n## Pooled confusion (rows=Qwen, cols=BERT)\n")
    lines.append("| Qwen \\ BERT | support | neutral | oppose |")
    lines.append("|---|---|---|---|")
    for i, s in enumerate(STANCE_ORDER):
        lines.append(f"| {s} | {cm[i,0]} | {cm[i,1]} | {cm[i,2]} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
