#!/usr/bin/env python3
"""Cross-cue transfer over the full 562-name bank (RQ3), from streamed features.

Reads the PCA-reduced features written by pipeline/13_stream_name_features.py and
reproduces, on 562 names instead of 12, everything the twelve-name pipeline reports:

  label_to_name    the transfer claim -- probe trained on explicit labels, applied
                   unchanged to name prompts
  control_*        the same probe fit to permuted training labels, then transferred
                   (Hewitt and Liang 2019), as a percentile band over permutations
  boot_*           95% cluster bootstrap over the 19 issues, matching the CI used
                   everywhere else in the thesis
  namboot_*        95% cluster bootstrap over the 562 *names* -- the interval the
                   twelve-name run could not produce, and the one that speaks to
                   whether the result is about a demographic category or about
                   particular strings

Balanced accuracy throughout. CPU only.

    python analysis/06_probe/transfer_namebank.py --tag llama \
        --feats /data/<user>/probe_activations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_identity_probe import RACE_GENDER


def fit_predict(Xtr, ytr, Xte):
    """Logistic head only: the scaler and PCA are already baked into the features."""
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def cluster_boot(y, pred, clusters, reps, rng, labels=RACE_GENDER):
    """Percentile band for balanced accuracy, resampling whole clusters.

    Balanced accuracy is the mean of the per-class recalls, and a recall is a ratio
    of two sums over rows. So a cluster only ever enters a resample through its own
    (hits, total) per class, and a resample is a weighted sum of those per-cluster
    pairs. Precomputing them turns each draw from a pass over 53,390 rows into a
    length-562 dot product, which is what makes 1000 draws x 65 layers tractable --
    the naive version was taking minutes per layer. Exact, not an approximation.
    """
    uniq, cidx = np.unique(clusters, return_inverse=True)
    K, C = len(labels), len(uniq)
    hits = np.zeros((C, K))
    total = np.zeros((C, K))
    for k, lab in enumerate(labels):
        is_k = (y == lab)
        np.add.at(total[:, k], cidx[is_k], 1.0)
        np.add.at(hits[:, k], cidx[is_k & (pred == y)], 1.0)

    # Multiplicity matrix: row b holds how often each cluster was drawn in draw b.
    take = rng.integers(0, C, size=(reps, C))
    M = np.zeros((reps, C))
    np.add.at(M, (np.repeat(np.arange(reps), C), take.ravel()), 1.0)

    num, den = M @ hits, M @ total
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(den > 0, num / den, np.nan)
    return np.nanmean(recall, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out-dir", default="results/probe_internal")
    ap.add_argument("--control-reps", type=int, default=10)
    ap.add_argument("--boot-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    f = Path(a.feats)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    exp = np.load(f / f"{a.tag}_explicit_feats.npz")
    nam = np.load(f / f"{a.tag}_namebank_feats.npz")
    emeta = pd.read_csv(f / f"{a.tag}_explicit_meta.csv")
    nmeta = pd.read_csv(f / f"{a.tag}_namebank_meta.csv")

    ye = emeta["cue_group"].to_numpy()
    yn = nmeta["cue_group"].to_numpy()
    issues = nmeta["issue_id"].to_numpy()
    names = nmeta["cue_value"].to_numpy()
    layers = sorted((k for k in exp.files), key=lambda k: int(k.split("_")[1]))
    print(f"[{a.tag}] {len(layers)} layers | explicit {len(ye)} | name {len(yn)} "
          f"({len(np.unique(names))} names, {len(np.unique(issues))} issues)", flush=True)

    rows = []
    for key in layers:
        li = int(key.split("_")[1])
        Xe, Xn = exp[key], nam[key]
        pred = fit_predict(Xe, ye, Xn)
        point = balanced_accuracy_score(yn, pred)

        ctrl = np.asarray([
            balanced_accuracy_score(yn, fit_predict(Xe, rng.permutation(ye), Xn))
            for _ in range(a.control_reps)])
        ib = cluster_boot(yn, pred, issues, a.boot_reps, rng)
        nb = cluster_boot(yn, pred, names, a.boot_reps, rng)

        rows.append({
            "layer": li, "chance": 1.0 / len(RACE_GENDER), "label_to_name": point,
            "boot_lo": float(np.percentile(ib, 2.5)), "boot_hi": float(np.percentile(ib, 97.5)),
            "nameboot_lo": float(np.percentile(nb, 2.5)),
            "nameboot_hi": float(np.percentile(nb, 97.5)),
            "control_mean": float(ctrl.mean()),
            "control_lo": float(np.percentile(ctrl, 2.5)),
            "control_hi": float(np.percentile(ctrl, 97.5)),
            "n_names": int(len(np.unique(names))), "n_rows": int(len(yn)),
        })
        print(f"  layer {li:3d}  transfer {point:.3f} "
              f"issue[{rows[-1]['boot_lo']:.3f},{rows[-1]['boot_hi']:.3f}] "
              f"name[{rows[-1]['nameboot_lo']:.3f},{rows[-1]['nameboot_hi']:.3f}] "
              f"ctrl {ctrl.mean():.3f}", flush=True)

    df = pd.DataFrame(rows)
    p = out_dir / f"{a.tag}_transfer_namebank.csv"
    df.to_csv(p, index=False)
    fin = df.iloc[-1]
    print(f"\nwrote {p}\n  final layer {fin.label_to_name:.3f} "
          f"[{fin.nameboot_lo:.3f}, {fin.nameboot_hi:.3f}] over names, "
          f"control {fin.control_mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
