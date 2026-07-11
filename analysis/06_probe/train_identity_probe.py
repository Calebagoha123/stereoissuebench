#!/usr/bin/env python3
"""Internal identity probes (arm B), trained locally on extracted activations.

Reads <tag>_acts.npz + <tag>_meta.csv from pipeline/08_extract_activations.py.

B1 — decodability. Per layer, an L2 logistic-regression probe decodes the cued
group from the residual stream, with GroupKFold over ISSUES (so it reads the
group, not the task). Done separately for: explicit demographic labels (race x
gender), implicit demographic NAMES (same 4 labels), explicit political (party),
implicit political (state class). The headline comparison is label-vs-name
decodability + a shuffled-label control for selectivity.

B1-transfer — the key test. Train the race x gender probe on EXPLICIT-label rows
and test it on NAME rows (and vice-versa). If the explicit "Black woman"
direction also fires for "My name is Aaliyah", the model internally represents
the name's demographic even though it never says so; if it sits at chance, the
name never reaches that representation.

B2 — political direction. Build a liberal<->conservative axis as the
Democrat-minus-Republican mean activation difference (per layer), then project
every group's activations onto it. Shows whether implicit cues move along the
political axis at all relative to baseline.

B3 — mediation. Correlate each group's political-axis projection shift (vs
baseline) against its ACTUAL written-stance shift (from bert_eval_*), across cue
groups: does internal political movement explain output movement, and do
names/states cluster near zero on both?

    python analysis/train_identity_probe.py --tag qwen \
        --acts <dir>/qwen_acts.npz --meta <dir>/qwen_meta.csv \
        --stance results/full/bert_eval_qwen.csv --out-dir results/probe_internal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("needs scikit-learn: pip install scikit-learn") from exc

# Probe = standardize -> top-PCA components -> L2 logistic regression. Reducing
# the 4096-d residual stream to its leading components keeps the decodable
# (linear) signal while making the per-layer CV sweep ~16x faster on CPU.
PCA_COMPONENTS = 256


def make_probe(n_train: int):
    n_comp = min(PCA_COMPONENTS, n_train - 1)
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp, random_state=0),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )


RACE_GENDER = ["black_woman", "black_man", "white_woman", "white_man"]
PARTY = ["democrat", "republican", "independent"]
STATE = ["blue_state", "red_state", "swing_state"]

# (subset name, cue_family filter, label set) for the decodability probes.
SUBSETS = [
    ("explicit_demographic", "explicit_demographic", RACE_GENDER),
    ("name", "implicit_demographic", RACE_GENDER),
    ("explicit_political", "explicit_political", PARTY),
    ("state", "implicit_political", STATE),
]


def load(acts_path: str, meta_path: str):
    npz = np.load(acts_path)
    layers = sorted((k for k in npz.files if k.startswith("layer_")),
                    key=lambda k: int(k.split("_")[1]))
    meta = pd.read_csv(meta_path)
    return npz, layers, meta


def layer_X(npz, layer) -> np.ndarray:
    """Float32 activations with any non-finite values (fp16-overflow legacy) zeroed."""
    return np.nan_to_num(npz[layer].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fit_eval_cv(X, y, groups, seed=0):
    """Balanced accuracy under GroupKFold (hold out whole issue groups)."""
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.empty_like(y)
    for tr, te in gkf.split(X, y, groups):
        clf = make_probe(len(tr))
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    return balanced_accuracy_score(y, preds)


def transfer_acc(Xtr, ytr, Xte, yte):
    clf = make_probe(len(Xtr))
    clf.fit(Xtr, ytr)
    return balanced_accuracy_score(yte, clf.predict(Xte))


def run_decodability(npz, layers, meta, rng) -> pd.DataFrame:
    rows = []
    for layer in layers:
        X_all = layer_X(npz, layer)
        li = int(layer.split("_")[1])
        for name, fam, labels in SUBSETS:
            m = meta["cue_family"].eq(fam) & meta["cue_group"].isin(labels)
            idx = np.where(m.to_numpy())[0]
            X = X_all[idx].astype(np.float32)
            y = meta.loc[m, "cue_group"].to_numpy()
            groups = meta.loc[m, "issue_id"].to_numpy()
            acc = fit_eval_cv(X, y, groups)
            # shuffled-label control (selectivity)
            y_shuf = rng.permutation(y)
            ctrl = fit_eval_cv(X, y_shuf, groups)
            rows.append({"layer": li, "subset": name, "n": len(idx),
                         "chance": 1.0 / len(labels),
                         "bal_acc": acc, "control_acc": ctrl,
                         "selectivity": acc - ctrl})
    return pd.DataFrame(rows)


def run_transfer(npz, layers, meta) -> pd.DataFrame:
    """Train race x gender on explicit labels, test on names (and reverse)."""
    exp = meta["cue_family"].eq("explicit_demographic") & meta["cue_group"].isin(RACE_GENDER)
    nam = meta["cue_family"].eq("implicit_demographic") & meta["cue_group"].isin(RACE_GENDER)
    ei, ni = np.where(exp.to_numpy())[0], np.where(nam.to_numpy())[0]
    ye, yn = meta.loc[exp, "cue_group"].to_numpy(), meta.loc[nam, "cue_group"].to_numpy()
    rows = []
    for layer in layers:
        li = int(layer.split("_")[1])
        XL = layer_X(npz, layer); Xe, Xn = XL[ei], XL[ni]
        rows.append({"layer": li, "chance": 0.25,
                     "label_to_name": transfer_acc(Xe, ye, Xn, yn),
                     "name_to_label": transfer_acc(Xn, yn, Xe, ye)})
    return pd.DataFrame(rows)


def run_political_axis(npz, layers, meta) -> tuple[pd.DataFrame, int]:
    """Dem-minus-Rep direction per layer; mean projection per cue group."""
    dem = (meta["cue_family"].eq("explicit_political") & meta["cue_group"].eq("democrat")).to_numpy()
    rep = (meta["cue_family"].eq("explicit_political") & meta["cue_group"].eq("republican")).to_numpy()
    base = (meta["cue_family"].eq("baseline")).to_numpy()
    group_keys = list(meta.groupby(["cue_family", "cue_group"]).groups.keys())

    rows, sep_by_layer = [], {}
    for layer in layers:
        li = int(layer.split("_")[1])
        X = layer_X(npz, layer)
        v = X[dem].mean(0) - X[rep].mean(0)
        v = v / (np.linalg.norm(v) + 1e-8)
        proj = X @ v
        base_mean = proj[base].mean()
        # separation = how far apart Dem and Rep sit on this axis (axis quality)
        sep_by_layer[li] = float(proj[dem].mean() - proj[rep].mean())
        for fam, grp in group_keys:
            mask = (meta["cue_family"].eq(fam) & meta["cue_group"].eq(grp)).to_numpy()
            rows.append({"layer": li, "cue_family": fam, "cue_group": grp,
                         "proj_mean": float(proj[mask].mean()),
                         "proj_shift": float(proj[mask].mean() - base_mean)})
    best_layer = max(sep_by_layer, key=sep_by_layer.get)
    return pd.DataFrame(rows), best_layer


def run_mediation(proj_df: pd.DataFrame, best_layer: int, stance_path: str) -> pd.DataFrame:
    stance = pd.read_csv(stance_path, usecols=["cue_family", "cue_group", "bert_liberal_score"],
                         low_memory=False)
    stance["bert_liberal_score"] = pd.to_numeric(stance["bert_liberal_score"], errors="coerce")
    base = stance.loc[stance["cue_family"].eq("baseline"), "bert_liberal_score"].mean()
    smean = (stance.groupby(["cue_family", "cue_group"])["bert_liberal_score"].mean()
             .rename("stance_mean").reset_index())
    smean["stance_shift"] = smean["stance_mean"] - base
    proj = proj_df[proj_df["layer"].eq(best_layer)][["cue_family", "cue_group", "proj_shift"]]
    out = proj.merge(smean, on=["cue_family", "cue_group"], how="inner")
    out = out[~out["cue_family"].eq("baseline")]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stance", help="bert_eval_<model>.csv for the B3 mediation link.")
    ap.add_argument("--out-dir", default="results/probe_internal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    npz, layers, meta = load(args.acts, args.meta)
    print(f"[{args.tag}] {len(layers)} layers, {len(meta)} rows")

    deco = run_decodability(npz, layers, meta, rng)
    deco.to_csv(out_dir / f"{args.tag}_decodability_by_layer.csv", index=False)
    transfer = run_transfer(npz, layers, meta)
    transfer.to_csv(out_dir / f"{args.tag}_cross_cue_transfer.csv", index=False)
    proj, best_layer = run_political_axis(npz, layers, meta)
    proj.to_csv(out_dir / f"{args.tag}_political_projection.csv", index=False)

    # headline summary: best-layer decodability per subset + best transfer
    best = (deco.sort_values("bal_acc").groupby("subset").tail(1)
            .set_index("subset")[["layer", "bal_acc", "control_acc", "selectivity"]])
    summary = {
        "tag": args.tag,
        "best_political_axis_layer": best_layer,
        "decodability_best": best.to_dict(orient="index"),
        "transfer_label_to_name_max": float(transfer["label_to_name"].max()),
        "transfer_name_to_label_max": float(transfer["name_to_label"].max()),
        "transfer_chance": 0.25,
    }
    if args.stance:
        med = run_mediation(proj, best_layer, args.stance)
        med.to_csv(out_dir / f"{args.tag}_mediation.csv", index=False)
        if len(med) > 2:
            r = float(np.corrcoef(med["proj_shift"], med["stance_shift"])[0, 1])
            summary["mediation_r_proj_vs_stance"] = r
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== Decodability (best layer per subset) ===")
    print(best.round(3).to_string())
    print(f"\nLabel->name transfer (max over layers): {summary['transfer_label_to_name_max']:.3f} "
          f"(chance 0.25)")
    if args.stance and "mediation_r_proj_vs_stance" in summary:
        print(f"Mediation r(proj_shift, stance_shift) = {summary['mediation_r_proj_vs_stance']:.3f}")
    print(f"\nWrote outputs to {out_dir}/{args.tag}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
