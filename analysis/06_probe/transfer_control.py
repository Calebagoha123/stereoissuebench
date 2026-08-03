#!/usr/bin/env python3
"""Shuffled-label control for the cross-cue transfer test (RQ3, main text).

`train_identity_probe.py` computes a shuffled-label control for the *decodability*
probes (train and test within one cue family) but not for the *transfer* probes
(train on explicit labels, test on names). The main text's transfer claim needs its
own control: a probe with identical capacity and identical training data, fit to
permuted labels, transferred unchanged to the name prompts. Anything the real probe
scores above this band is signal shared between the two cue families rather than
probe capacity or class structure.

Control task in the sense of Hewitt and Liang (2019), matching the decodability
control already reported, but applied across cue families.

Repeats the permutation `--reps` times per layer and reports mean plus a percentile
band, so the figure can shade the control the way Neplenbroek et al. (arXiv:2505.16467)
Figure 2 does.

CPU only -- reads the cached activations, fits no new model on GPU.

    python analysis/06_probe/transfer_control.py --tag llama \
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-dir", default="results/probe_internal")
    ap.add_argument("--reps", type=int, default=10)
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
    print(f"[{a.tag}] {len(layers)} layers | explicit n={len(ei)} name n={len(ni)} "
          f"| {a.reps} permutations/layer", flush=True)

    rows = []
    for layer in layers:
        li = int(layer.split("_")[1])
        XL = layer_X(npz, layer)
        Xe, Xn = XL[ei], XL[ni]
        accs = []
        for _ in range(a.reps):
            # Permute the *training* labels only; the name-side truth is untouched,
            # so the probe keeps its capacity and loses only the label-activation link.
            y_shuf = rng.permutation(ye)
            accs.append(balanced_accuracy_score(yn, transfer_pred(Xe, y_shuf, Xn)))
        accs = np.asarray(accs)
        rows.append({"layer": li, "chance": 0.25,
                     "control_mean": accs.mean(),
                     "control_lo": np.percentile(accs, 2.5),
                     "control_hi": np.percentile(accs, 97.5),
                     "control_max": accs.max(), "reps": a.reps})
        print(f"  layer {li:3d}  control {accs.mean():.3f} "
              f"[{np.percentile(accs, 2.5):.3f}, {np.percentile(accs, 97.5):.3f}]", flush=True)

    df = pd.DataFrame(rows)
    p = out_dir / f"{a.tag}_transfer_control.csv"
    df.to_csv(p, index=False)
    print(f"\nwrote {p}\n  mean control over layers: {df.control_mean.mean():.3f} "
          f"(chance 0.25)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
