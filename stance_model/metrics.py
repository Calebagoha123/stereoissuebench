"""Stance-regression metrics shared by training and evaluation.

The model predicts writer_stance on a 0-100 scale (0 = opposes proposition,
50 = neutral, 100 = supports). We report:
  - spearman / pearson: rank + linear agreement with the human reader mean
    (spearman is the primary, scale-free metric).
  - mae / rmse: error on the 0-100 scale.
  - macro_f1_3: a 3-way support/neutral/oppose macro-F1, comparable in spirit to
    IssueBench's stance F1. Thresholds are a reporting choice, not baked into the
    model; default neutral band is [NEU_LO, NEU_HI].
"""

from __future__ import annotations

import numpy as np

NEU_LO = 40.0
NEU_HI = 60.0


def to_bins(stance: np.ndarray, lo: float = NEU_LO, hi: float = NEU_HI) -> np.ndarray:
    """0=oppose (<lo), 1=neutral ([lo,hi]), 2=support (>hi)."""
    bins = np.ones_like(stance, dtype=int)
    bins[stance < lo] = 0
    bins[stance > hi] = 2
    return bins


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def stance_metrics(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    gold = np.asarray(gold, dtype=float)
    err = pred - gold
    return {
        "n": int(len(gold)),
        "spearman": _spearman(pred, gold),
        "pearson": float(np.corrcoef(pred, gold)[0, 1]),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "macro_f1_3": _macro_f1(to_bins(gold), to_bins(pred)),
        "bin_accuracy": float(np.mean(to_bins(gold) == to_bins(pred))),
    }
