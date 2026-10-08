#!/usr/bin/env python3
"""Cluster-bootstrap confidence band for the cross-cue transfer curve (RQ3).

`transfer_control.py` shades the *null* -- what a same-capacity probe scores when
the label-activation link is broken. That band says nothing about how precisely the
real transfer curve is estimated, so the figure currently draws a bare line against
a shaded null. This script supplies the matching band for the line itself.

The probe is fit once per layer on the explicit-label prompts (exactly as in
`train_identity_probe.run_transfer`) and applied once to the name prompts; the
bootstrap then resamples the *evaluation* set. Resampling is over issues, not over
individual prompts: the four race x gender name conditions are crossed with the same
issue set, so prompt-level resampling would treat four responses to one issue as four
independent draws and understate the spread. Refitting is not repeated -- the
quantity being bounded is the accuracy of a fixed probe on a resampled name corpus,
which is what the transfer claim asserts.

Balanced accuracy throughout, matching the curve it bounds.

    python analysis/06_probe/transfer_ci.py --tag llama \
        --acts /data/kell8360/probe_activations/llama_acts.npz \
        --meta /data/kell8360/probe_activations/llama_meta.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_identity_probe import RACE_GENDER, layer_X, load, transfer_pred


def cluster_boot(y, pred, clusters, reps, rng):
    """Percentile band for balanced accuracy, resampling whole issue clusters."""
    uniq = np.unique(clusters)
    idx_by_cluster = [np.where(clusters == c)[0] for c in uniq]
    accs = []
    for _ in range(reps):
        take = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_cluster[t] for t in take])
        yb = y[idx]
        # A resample can drop a class entirely; balanced accuracy is then the mean
        # over the classes that survive, which is the right conditional quantity.
        accs.append(balanced_accuracy_score(yb, pred[idx]))
    return np.asarray(accs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-dir", default="results/probe_internal")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    npz, layers, meta = load(a.acts, a.meta)

    exp = meta["cue_family"].eq("explicit_demographic") & meta["cue_group"].isin(RACE_GENDER)
    nam = meta["cue_family"].eq("implicit_demographic") & meta["cue_group"].isin(RACE_GENDER)
    ei, ni = np.where(exp.to_numpy())[0], np.where(nam.to_numpy())[0]
    ye = meta.loc[exp, "cue_group"].to_numpy()
    yn = meta.loc[nam, "cue_group"].to_numpy()
    clusters = meta.loc[nam, "issue_id"].to_numpy()
    print(f"[{a.tag}] {len(layers)} layers | explicit n={len(ei)} name n={len(ni)} "
          f"| {len(np.unique(clusters))} issue clusters | {a.reps} bootstrap reps",
          flush=True)

    rows = []
    for layer in layers:
        li = int(layer.split("_")[1])
        XL = layer_X(npz, layer)
        pred_n = transfer_pred(XL[ei], ye, XL[ni])
        point = balanced_accuracy_score(yn, pred_n)
        accs = cluster_boot(yn, pred_n, clusters, a.reps, rng)
        rows.append({"layer": li, "chance": 0.25,
                     "label_to_name": point,
                     "boot_lo": float(np.percentile(accs, 2.5)),
                     "boot_hi": float(np.percentile(accs, 97.5)),
                     "boot_se": float(accs.std(ddof=1)), "reps": a.reps})
        print(f"  layer {li:3d}  transfer {point:.3f} "
              f"[{rows[-1]['boot_lo']:.3f}, {rows[-1]['boot_hi']:.3f}]", flush=True)

    df = pd.DataFrame(rows)
    p = out_dir / f"{a.tag}_transfer_ci.csv"
    df.to_csv(p, index=False)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
