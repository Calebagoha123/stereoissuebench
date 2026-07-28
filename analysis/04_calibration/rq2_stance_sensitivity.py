#!/usr/bin/env python3
"""Stance-reduction sensitivity for the RQ2 conclusions (checks B and C).

The headline maps the classifier-of-record (luna) 0-100 stance to {-1,0,+1} with a
neutral band of pred in [40,60] (cutoff 50 +/- 10), then averages. Two objections:

  B. Threshold sensitivity — "your effects are an artifact of an arbitrary
     cutoff." We recompute every Delta_k and the calibration slope under neutral
     half-bands h in {5,10(default),15,20} (i.e. +/-0.05..0.20 on the 0-1 score).

  C. Directional-only reanalysis — "your under-personalisation is an artifact of
     letting the model say 'it depends' when CES respondents could not." We drop
     neutrals entirely and average the liberal score over directional responses
     only (the forced-choice analogue of the CES scale), then refit the slope.

For each variant we report the three load-bearing conclusions:
  (1) Republican is the strongest cue, (2) the name cues are ~null,
  (3) the calibration slope is < 1.

Reads results/full_3x/{luna,bert}_eval_*.csv (scorer of record; SCORER env toggles)
+ results/full_3x/ces_estimates.csv. Writes results/robustness/stance_sensitivity.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, CUE_ORDER, FULL3X, ROBUST, EVAL_PREFIX, _parse_prompt_id
from _regression import ols_through_origin, deming

# raw 0-100 stance column matching the active scorer ("luna_eval" -> "luna_pred_stance")
RAW_SCORE_COL = f"{EVAL_PREFIX.split('_')[0]}_pred_stance"


def load_raw(model):
    df = pd.read_csv(FULL3X / f"{EVAL_PREFIX}_{model}.csv", low_memory=False,
                     usecols=["prompt_id", "arm", "cue_family", "cue_group",
                              "ces_variable", "liberal_sign", RAW_SCORE_COL])
    meta = _parse_prompt_id(df["prompt_id"])
    df["template_id"] = meta["template_id"]
    df["model"] = model
    return df


def liberal_from_band(pred, sign, h):
    """{-1,0,+1} liberal score under neutral half-band h around 50."""
    s = np.where(pred > 50 + h, 1, np.where(pred < 50 - h, -1, 0))
    return s * sign


def shift_and_slope(df, ces, mode):
    """Return per-(model,cue) shift + pooled/per-model slopes for one reduction."""
    recs = []
    for model in MODELS:
        d = df[df.model == model]
        base_all = d[d.cue_family == "baseline"]
        for fam, grp in CUE_ORDER:
            cue = d[(d.cue_family == fam) & (d.cue_group == grp)]
            if cue.empty:
                continue
            base = (base_all[base_all.template_id.isin(cue.template_id.unique())]
                    if fam.startswith("implicit") else base_all)
            if mode == "directional":
                cval = cue.loc[cue.lib != 0, "lib"]
                bval = base.loc[base.lib != 0, "lib"]
            else:
                cval, bval = cue["lib"], base["lib"]
            recs.append({"model": model, "cue_family": fam, "cue_group": grp,
                         "shift": cval.mean() - bval.mean()})
    mt = pd.DataFrame(recs).merge(
        ces[["cue_family", "cue_group", "ces_shift_mean"]], on=["cue_family", "cue_group"])
    # slopes
    out_slopes = {}
    for scope in ["pooled"] + MODELS:
        sub = mt if scope == "pooled" else mt[mt.model == scope]
        x, y = sub["ces_shift_mean"].to_numpy(), sub["shift"].to_numpy()
        bo, _ = ols_through_origin(x, y)
        # equal-error Deming (delta=1) since variances aren't recomputed per variant
        bd, _ = deming(x, y, 1.0)
        out_slopes[scope] = (bo, bd)
    return mt, out_slopes


def main():
    ces = pd.read_csv(FULL3X / "ces_estimates.csv")
    raw = pd.concat([load_raw(m) for m in MODELS], ignore_index=True)

    variants = [("band_h05", "band", 5), ("band_h10_default", "band", 10),
                ("band_h15", "band", 15), ("band_h20", "band", 20),
                ("directional_only", "directional", 10)]

    rows = []
    slope_rows = []
    for name, mode, h in variants:
        raw["lib"] = liberal_from_band(raw[RAW_SCORE_COL].to_numpy(),
                                       raw["liberal_sign"].to_numpy(), h)
        mt, slopes = shift_and_slope(raw, ces, mode)
        mt["variant"] = name
        rows.append(mt)
        # conclusion checks (pooled over models: mean shift per cue)
        pooled = mt.groupby(["cue_family", "cue_group"])["shift"].mean()
        rep = pooled[("explicit_political", "republican")]
        strongest = pooled.abs().idxmax()
        names = [pooled[("implicit_demographic", g)] for g in
                 ["black_woman", "black_man", "white_woman", "white_man"]]
        slope_rows.append({
            "variant": name,
            "slope_pooled_OLSorigin": slopes["pooled"][0],
            "slope_pooled_Deming": slopes["pooled"][1],
            "slope_llama": slopes["llama"][0], "slope_gemma": slopes["gemma"][0],
            "slope_qwen": slopes["qwen"][0],
            "republican_shift": rep,
            "republican_is_strongest": strongest == ("explicit_political", "republican"),
            "max_abs_name_shift": max(abs(n) for n in names),
            "slope_below_1": slopes["pooled"][0] < 1 and slopes["pooled"][1] < 1,
        })

    pd.concat(rows, ignore_index=True).to_csv(ROBUST / "stance_sensitivity_shifts.csv", index=False)
    sl = pd.DataFrame(slope_rows)
    sl.to_csv(ROBUST / "stance_sensitivity.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== Stance-reduction sensitivity (checks B threshold + C directional-only) ===\n")
    for _, r in sl.iterrows():
        print(f"[{r['variant']:>18}]  slope OLS={r['slope_pooled_OLSorigin']:.3f} "
              f"Deming={r['slope_pooled_Deming']:.3f}  "
              f"(L={r['slope_llama']:.2f} G={r['slope_gemma']:.2f} Q={r['slope_qwen']:.2f})")
        print(f"{'':22} Republican shift={r['republican_shift']:+.3f} "
              f"strongest={r['republican_is_strongest']}  "
              f"max|name|={r['max_abs_name_shift']:.3f}  slope<1={r['slope_below_1']}")
    print("\nConclusions hold across all variants:")
    print(f"  Republican strongest cue:  {sl['republican_is_strongest'].all()}")
    print(f"  Names null (max|shift|<0.05 every variant): {(sl['max_abs_name_shift']<0.05).all()}")
    print(f"  Slope < 1 every variant:   {sl['slope_below_1'].all()}")
    print(f"\nWrote {ROBUST/'stance_sensitivity.csv'}")


if __name__ == "__main__":
    main()
