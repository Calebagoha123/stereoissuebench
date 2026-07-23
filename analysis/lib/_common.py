#!/usr/bin/env python3
"""Shared loaders + the model-side shift estimator for the RQ2 robustness suite.

Everything downstream (Deming/free-intercept regression, TOST, composition
flattening, DiD variance propagation, refusal bounds) reads the per-(model, cue)
model-shift table produced here, so the estimator is defined in exactly one place.

Data of record
--------------
results/full_3x/bert_eval_{llama,gemma,qwen}.csv — the fresh 3-repeat 2k-token
rerun, DeBERTa stance labels (``bert_liberal_score`` in {-1, 0, +1}; +1 = writes
the liberal side of the issue, -1 = the conservative side, 0 = neutral).

Estimator
---------
For a cue k, the model shift is

    Delta_k = mean(bert_liberal_score | cue k) - mean(bert_liberal_score | baseline)

The **baseline** is the no-cue Arm-A condition. Arm-B cues (rotated names/states)
were generated on the genre-proportional ~35-template subset, so for Arm-B cues
the baseline is *template-matched* to that same subset (otherwise the delta would
confound the cue with the 145-vs-35 template composition). Arm-A cues use the
full 145-template baseline.

Uncertainty is an **issue-clustered nonparametric bootstrap**: we resample the 19
issues with replacement (the issue is the primary unit of correlated variance and
matches the issue-averaged CES estimator), recompute the cued and baseline means
over the resampled issues, and take the difference. This yields both a bootstrap
CI and a bootstrap variance for each Delta_k, the latter feeding the Deming
regression and the DiD variance propagation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 3 open-source (3-rep) + 2 frontier (1-rep, API) models. The frontier arm is
# single-generation, so rep-dependent checks (generation_variance) skip it; all
# rep-agnostic RQ2 tables (model_shift, calibration, composition) include it.
MODELS = ["llama", "gemma", "qwen", "gpt56terra", "sonnet5"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B", "qwen": "Qwen3.6-27B",
               "gpt56terra": "GPT-5.6 Terra", "sonnet5": "Claude Sonnet 5"}
MODEL_COLOUR = {"llama": "#2e6da4", "gemma": "#27915b", "qwen": "#c0392b",
                "gpt56terra": "#8e44ad", "sonnet5": "#2980b9"}

FULL3X = Path("results/full_3x")
ROBUST = Path("results/robustness")

# The 14 non-baseline cues, in a stable display order, with their CES x-partner.
CUE_ORDER = [
    ("explicit_political", "democrat"),
    ("explicit_political", "republican"),
    ("explicit_political", "independent"),
    ("explicit_demographic", "black_woman"),
    ("explicit_demographic", "black_man"),
    ("explicit_demographic", "white_woman"),
    ("explicit_demographic", "white_man"),
    ("implicit_political", "blue_state"),
    ("implicit_political", "red_state"),
    ("implicit_political", "swing_state"),
    ("implicit_demographic", "black_woman"),
    ("implicit_demographic", "black_man"),
    ("implicit_demographic", "white_woman"),
    ("implicit_demographic", "white_man"),
]

CUE_DISPLAY = {
    ("explicit_political", "democrat"): "Democrat (label)",
    ("explicit_political", "republican"): "Republican (label)",
    ("explicit_political", "independent"): "Independent (label)",
    ("explicit_demographic", "black_woman"): "Black woman (label)",
    ("explicit_demographic", "black_man"): "Black man (label)",
    ("explicit_demographic", "white_woman"): "White woman (label)",
    ("explicit_demographic", "white_man"): "White man (label)",
    ("implicit_political", "blue_state"): "Blue state",
    ("implicit_political", "red_state"): "Red state",
    ("implicit_political", "swing_state"): "Swing state",
    ("implicit_demographic", "black_woman"): "“Black-female” name",
    ("implicit_demographic", "black_man"): "“Black-male” name",
    ("implicit_demographic", "white_woman"): "“White-female” name",
    ("implicit_demographic", "white_man"): "“White-male” name",
}


def _parse_prompt_id(pid: pd.Series) -> pd.DataFrame:
    """prompt_id = issue__template__cue__rep ; cue may hold the Arm-B instance."""
    parts = pid.str.split("__", expand=True)
    out = pd.DataFrame(index=pid.index)
    out["template_id"] = parts[1]
    out["rep"] = parts[3]
    # Arm-B instance = trailing token of the cue segment after the cue_group.
    out["cue_seg"] = parts[2]
    return out


def load_model(model: str, in_dir: Path = FULL3X) -> pd.DataFrame:
    """Long per-response frame: model, arm, cue_family, cue_group, issue_id,
    template_id, instance, rep, y (bert_liberal_score as float)."""
    df = pd.read_csv(in_dir / f"bert_eval_{model}.csv", low_memory=False)
    meta = _parse_prompt_id(df["prompt_id"])
    df = pd.concat([df, meta], axis=1)
    df["model"] = model
    df["y"] = df["bert_liberal_score"].astype(float)
    # Arm-B instance name: strip the "<family>_<group>_" prefix from the cue seg.
    def _instance(row):
        if row["arm"] != "B":
            return ""
        prefix = f"{row['cue_family']}_{row['cue_group']}_"
        seg = row["cue_seg"]
        return seg[len(prefix):] if seg.startswith(prefix) else seg
    df["instance"] = df.apply(_instance, axis=1)
    keep = ["model", "arm", "cue_family", "cue_group", "issue_id", "ces_variable",
            "stance_target", "liberal_sign", "template_id", "instance", "rep",
            "bert_collapsed_stance", "y"]
    return df[keep].copy()


def load_all(in_dir: Path = FULL3X) -> pd.DataFrame:
    return pd.concat([load_model(m, in_dir) for m in MODELS], ignore_index=True)


def _mean_over_issues(sub: pd.DataFrame, issues: np.ndarray) -> float:
    """Grand mean of y over the given (bootstrap-resampled) issue list.

    Rows are pooled across the resampled issues with multiplicity, so an issue
    drawn twice counts twice — a standard clustered bootstrap.
    """
    g = sub.groupby("issue_id")["y"]
    sums = g.sum()
    counts = g.count()
    s = c = 0.0
    for iss in issues:
        if iss in sums.index:
            s += sums[iss]
            c += counts[iss]
    return s / c if c else np.nan


def shift_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 20260711) -> pd.DataFrame:
    """Per-(model, cue) Delta_k with issue-clustered bootstrap CI + variance.

    Arm-B cues are compared to a template-matched baseline (baseline rows whose
    template_id appears in that cue's Arm-B template subset).
    """
    rng = np.random.default_rng(seed)
    recs = []
    for model in MODELS:
        d = df[df["model"] == model]
        base_all = d[d["cue_family"] == "baseline"]
        issues = np.sort(d["issue_id"].unique())
        n_iss = len(issues)
        # Pre-draw one issue bootstrap design shared across cues within a model so
        # cued/baseline deltas use the same resampled issues (paired).
        boot_idx = rng.integers(0, n_iss, size=(n_boot, n_iss))
        for fam, grp in CUE_ORDER:
            cue = d[(d["cue_family"] == fam) & (d["cue_group"] == grp)]
            if cue.empty:
                continue
            if fam.startswith("implicit"):
                tmpl = cue["template_id"].unique()
                base = base_all[base_all["template_id"].isin(tmpl)]
            else:
                base = base_all
            point = cue["y"].mean() - base["y"].mean()
            boots = np.empty(n_boot)
            for b in range(n_boot):
                iss = issues[boot_idx[b]]
                boots[b] = _mean_over_issues(cue, iss) - _mean_over_issues(base, iss)
            boots = boots[~np.isnan(boots)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            recs.append({
                "model": model, "cue_family": fam, "cue_group": grp,
                "cue_display": CUE_DISPLAY[(fam, grp)],
                "n_cue": len(cue), "n_base": len(base),
                "model_shift": point,
                "model_shift_var": float(np.var(boots, ddof=1)),
                "model_shift_se": float(np.std(boots, ddof=1)),
                "model_shift_lo": float(lo), "model_shift_hi": float(hi),
            })
    return pd.DataFrame(recs)


def merged_table(n_boot: int = 2000) -> pd.DataFrame:
    """Model shifts joined to CES shifts (x) — the input to the RQ2 regression."""
    df = load_all()
    mt = shift_table(df, n_boot=n_boot)
    ces = pd.read_csv(FULL3X / "ces_estimates.csv")
    ces = ces[["cue_family", "cue_group", "cue_label", "ces_score_mean",
               "ces_population_mean", "ces_shift_mean", "ces_shift_ci_low",
               "ces_shift_ci_high", "subgroup_n", "ces_n_issues"]]
    out = mt.merge(ces, on=["cue_family", "cue_group"], how="left")
    return out


if __name__ == "__main__":
    ROBUST.mkdir(parents=True, exist_ok=True)
    tbl = merged_table(n_boot=2000)
    out = ROBUST / "model_shift_table.csv"
    tbl.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(tbl[["model", "cue_display", "model_shift", "model_shift_se",
               "model_shift_lo", "model_shift_hi", "ces_shift_mean"]].to_string(index=False))
    print(f"\nWrote {out}  ({len(tbl)} rows)")
