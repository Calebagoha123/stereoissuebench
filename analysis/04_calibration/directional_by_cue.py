#!/usr/bin/env python3
"""Directional-only calibration, disaggregated by cue (thesis revision audit).

The pooled directional-only slope (~1.0, tab:threshold) must not be read as
magnitude calibration: it pools a Republican overshoot against a Democrat
attenuation. This script makes that decomposition explicit from the existing
per-response luna scores (no new generations):

1. per-cue directional-only shifts (pooled over models and per model) next to
   the CES subgroup shift, with the overshoot ratio shift/ces;
2. pooled + per-model directional-only slopes with and without the three
   explicit party cues (OLS-through-origin, matching tab:threshold);
3. the committed-response baseline mean per model (the "already liberal
   committed baseline" against which the Democrat attenuation happens).

Reads results/robustness/luna_calibration_shifts.csv (variant luna_directional)
+ the luna_eval per-response files for the committed baselines.
Writes results/robustness/directional_by_cue.csv and
results/robustness/directional_slopes_no_party.csv.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, ROBUST, FULL3X  # noqa: E402
from _regression import ols_through_origin, deming  # noqa: E402

PARTY = ("explicit_political",)


def slope_rows(mt: pd.DataFrame) -> list[dict]:
    rows = []
    for scope in ["pooled"] + MODELS:
        sub = mt if scope == "pooled" else mt[mt.model == scope]
        for label, s in [("all", sub), ("no_party", sub[~sub.cue_family.isin(PARTY)])]:
            x, y = s["ces_shift_mean"].to_numpy(), s["shift"].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            bo, se = ols_through_origin(x[m], y[m])
            bd, _ = deming(x[m], y[m], 1.0)
            rows.append({"scope": scope, "cues": label, "n": int(m.sum()),
                         "slope_ols_origin": bo, "slope_se": se,
                         "slope_deming_delta1": bd})
    return rows


def committed_baseline() -> pd.DataFrame:
    recs = []
    for model in MODELS:
        df = pd.read_csv(FULL3X / f"luna_eval_{model}.csv", low_memory=False,
                         usecols=["cue_family", "luna_pred_stance", "liberal_sign"])
        b = df[df.cue_family == "baseline"]
        disc = np.where(b["luna_pred_stance"] > 60, 1,
                        np.where(b["luna_pred_stance"] < 40, -1, 0)) * b["liberal_sign"]
        directional = disc[disc != 0]
        recs.append({"model": model,
                     "baseline_mean_all": float(disc.mean()),
                     "baseline_mean_directional": float(directional.mean()),
                     "n_directional": int((disc != 0).sum()), "n_all": len(disc)})
    return pd.DataFrame(recs)


def main() -> int:
    s = pd.read_csv(ROBUST / "luna_calibration_shifts.csv")
    d = s[s.variant == "luna_directional"].copy()

    # 1. per-cue table: pooled over models + per model, with overshoot ratio
    pooled = (d.groupby(["cue_family", "cue_group"])
                .agg(shift_pooled=("shift", "mean"), ces=("ces_shift_mean", "first"))
                .reset_index())
    pooled["ratio"] = pooled["shift_pooled"] / pooled["ces"]
    per_model = d.pivot_table(index=["cue_family", "cue_group"], columns="model",
                              values="shift").reset_index()
    out = pooled.merge(per_model, on=["cue_family", "cue_group"])
    out.to_csv(ROBUST / "directional_by_cue.csv", index=False)

    # 2. slopes with/without party cues
    sl = pd.DataFrame(slope_rows(d))
    sl.to_csv(ROBUST / "directional_slopes_no_party.csv", index=False)

    # 3. committed baselines
    cb = committed_baseline()
    cb.to_csv(ROBUST / "directional_baselines.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== Directional-only shifts by cue (pooled over models) ===")
    print(out.round(3).to_string(index=False))
    print("\n=== Directional-only slopes, with vs without party cues ===")
    print(sl.round(3).to_string(index=False))
    print("\n=== Committed-response baseline means ===")
    print(cb.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
