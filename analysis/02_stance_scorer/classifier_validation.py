#!/usr/bin/env python3
"""Classifier validation (check A, the cheap parts): 3-class confusion matrix,
per-class precision/recall/F1, and borderline (neutral-band) accuracy for the
DeBERTa stance scorer, from its cross-validated out-of-fold predictions.

The headline rests on this one scorer, so we document its error structure. Truth
= held-out human ``writer_stance`` (0-100); prediction = cross-validated
``pred_stance``. Both collapsed to {oppose, neutral, support} with the same
[40,60] neutral band the pipeline uses. Complements cv_metrics.json (which already
reports Spearman 0.93, macro-F1 0.84, binary accuracy 0.89).

Reads cv_oof_predictions.csv. Writes results/robustness/classifier_confusion.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import ROBUST

LABELS = ["oppose", "neutral", "support"]


def collapse(s, h=10):
    return np.where(s > 50 + h, "support", np.where(s < 50 - h, "oppose", "neutral"))


def main():
    d = pd.read_csv("results/stance_model_cv/cv_oof_predictions.csv")
    d["true3"] = collapse(d["writer_stance"].to_numpy())
    d["pred3"] = collapse(d["pred_stance"].to_numpy())

    cm = pd.crosstab(d["true3"], d["pred3"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    cm.to_csv(ROBUST / "classifier_confusion.csv")

    print("=== DeBERTa stance scorer validation (cross-validated OOF, n={}) ===\n".format(len(d)))
    print("Confusion matrix (rows = human truth, cols = predicted):")
    print(cm.to_string())
    print()
    # per-class precision/recall/F1
    print(f"{'class':>10} {'prec':>6} {'recall':>7} {'F1':>6} {'support':>8}")
    f1s = []
    for c in LABELS:
        tp = cm.loc[c, c]
        prec = tp / cm[c].sum() if cm[c].sum() else 0
        rec = tp / cm.loc[c].sum() if cm.loc[c].sum() else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f1s.append(f1)
        print(f"{c:>10} {prec:6.3f} {rec:7.3f} {f1:6.3f} {int(cm.loc[c].sum()):8d}")
    acc = np.mean(d["true3"].to_numpy() == d["pred3"].to_numpy())
    print(f"\n  macro-F1(3) = {np.mean(f1s):.3f}   3-class accuracy = {acc:.3f}")

    # borderline: accuracy inside vs outside the neutral band on the TRUTH scale
    near = d[(d["writer_stance"] > 40) & (d["writer_stance"] < 60)]
    far = d[(d["writer_stance"] <= 40) | (d["writer_stance"] >= 60)]
    print(f"\nBorderline analysis (where errors live):")
    print(f"  truth in neutral band [40,60]:  n={len(near):5d}  3-class acc={np.mean(near['true3']==near['pred3']):.3f}")
    print(f"  truth outside band (clear side): n={len(far):5d}  3-class acc={np.mean(far['true3']==far['pred3']):.3f}")
    print(f"  => directional errors (support<->oppose confusions): "
          f"{int(cm.loc['support','oppose']+cm.loc['oppose','support'])} / {len(d)} "
          f"({100*(cm.loc['support','oppose']+cm.loc['oppose','support'])/len(d):.2f}%)")
    print(f"\nWrote {ROBUST/'classifier_confusion.csv'}")


if __name__ == "__main__":
    main()
