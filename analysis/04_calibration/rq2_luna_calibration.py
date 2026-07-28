#!/usr/bin/env python3
"""RQ2 CES-calibration slope: DeBERTa vs luna (discretized) vs luna (continuous).

The neutral-compression objection to the DeBERTa headline is that DeBERTa maps
~48-58% of responses to neutral (vs 25-55% for luna), shrinking every model shift
toward zero and pulling the CES-calibration slope below 1. This script re-fits the
slope with the classifier of record (GPT-5.6 luna) under matched reductions so we
can see whether the compression fix lifts the slope to ~1.0 -- which would let us
report an absolute CES calibration and retire the directional-only reframe.

Reductions (all share the [40,60] neutral band == stance_model/metrics.py):
  deberta_band10        DeBERTa {-1,0,+1}, keep neutrals            (the headline)
  deberta_directional   DeBERTa {-1,0,+1}, drop neutrals           (the reframe)
  luna_disc10           luna    {-1,0,+1}, keep neutrals           (like-for-like)
  luna_directional      luna    {-1,0,+1}, drop neutrals
  luna_cont             luna    ((s-50)/50)*sign continuous, keep all   (the fix)

Model shift per (model, cue) = mean(y | cue) - mean(y | baseline); Arm-B implicit
cues use a template-matched baseline (same logic as _common.shift_table). Pooled
+ per-model slopes are OLS-through-origin and equal-error Deming (delta=1).

Reads results/full_3x/{bert_eval,luna_eval}_*.csv + ces_estimates.csv.
Writes results/robustness/luna_calibration_{shifts,slopes}.csv.
"""
from __future__ import annotations

import sys
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, CUE_ORDER, FULL3X, ROBUST, _parse_prompt_id  # noqa: E402
from _regression import ols_through_origin, deming  # noqa: E402

NEU_LO, NEU_HI = 40.0, 60.0


def _load(model: str, source: str) -> pd.DataFrame:
    """Per-response frame with model, cue keys, template_id, raw score + sign."""
    if source == "deberta":
        f, col = FULL3X / f"bert_eval_{model}.csv", "bert_pred_stance"
    else:
        f, col = FULL3X / f"luna_eval_{model}.csv", "luna_pred_stance"
    df = pd.read_csv(f, low_memory=False,
                     usecols=["prompt_id", "cue_family", "cue_group",
                              "liberal_sign", col])
    df = df.rename(columns={col: "raw"})
    df["template_id"] = _parse_prompt_id(df["prompt_id"])["template_id"]
    df["model"] = model
    return df


def reduce(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Attach y (value to average) and keep (row included in the mean)."""
    raw, sign = df["raw"].to_numpy(), df["liberal_sign"].to_numpy()
    disc = np.where(raw > NEU_HI, 1, np.where(raw < NEU_LO, -1, 0)) * sign
    df = df.copy()
    if mode == "cont":
        df["y"] = ((raw - 50.0) / 50.0) * sign
        df["keep"] = True
    elif mode == "directional":
        df["y"] = disc.astype(float)
        df["keep"] = disc != 0
    else:  # band
        df["y"] = disc.astype(float)
        df["keep"] = True
    return df


def shifts(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for model in MODELS:
        d = df[(df.model == model) & df.keep]
        base_all = d[d.cue_family == "baseline"]
        for fam, grp in CUE_ORDER:
            cue = d[(d.cue_family == fam) & (d.cue_group == grp)]
            if cue.empty:
                continue
            base = (base_all[base_all.template_id.isin(cue.template_id.unique())]
                    if fam.startswith("implicit") else base_all)
            recs.append({"model": model, "cue_family": fam, "cue_group": grp,
                         "shift": cue["y"].mean() - base["y"].mean()})
    return pd.DataFrame(recs)


def slopes(mt: pd.DataFrame) -> dict:
    out = {}
    for scope in ["pooled"] + MODELS:
        sub = mt if scope == "pooled" else mt[mt.model == scope]
        x, y = sub["ces_shift_mean"].to_numpy(), sub["shift"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            out[scope] = (np.nan, np.nan)
            continue
        bo, _ = ols_through_origin(x[m], y[m])
        bd, _ = deming(x[m], y[m], 1.0)
        out[scope] = (bo, bd)
    return out


def main() -> int:
    ROBUST.mkdir(parents=True, exist_ok=True)
    ces = pd.read_csv(FULL3X / "ces_estimates.csv")[
        ["cue_family", "cue_group", "ces_shift_mean"]]

    raw = {src: pd.concat([_load(m, src) for m in MODELS], ignore_index=True)
           for src in ("deberta", "luna")}

    variants = [
        ("deberta_band10", "deberta", "band"),
        ("deberta_directional", "deberta", "directional"),
        ("luna_disc10", "luna", "band"),
        ("luna_directional", "luna", "directional"),
        ("luna_cont", "luna", "cont"),
    ]

    all_shifts, slope_rows = [], []
    for name, src, mode in variants:
        mt = shifts(reduce(raw[src], mode)).merge(ces, on=["cue_family", "cue_group"])
        mt["variant"] = name
        all_shifts.append(mt)
        s = slopes(mt)
        pooled = mt.groupby(["cue_family", "cue_group"])["shift"].mean()
        names = [abs(pooled[("implicit_demographic", g)]) for g in
                 ["black_woman", "black_man", "white_woman", "white_man"]]
        slope_rows.append({
            "variant": name,
            "slope_pooled_OLS": s["pooled"][0], "slope_pooled_Deming": s["pooled"][1],
            "slope_llama": s["llama"][0], "slope_gemma": s["gemma"][0],
            "slope_qwen": s["qwen"][0], "slope_gpt56terra": s["gpt56terra"][0],
            "slope_sonnet5": s["sonnet5"][0],
            "republican_shift": pooled[("explicit_political", "republican")],
            "max_abs_name_shift": max(names),
            "slope_below_1": s["pooled"][0] < 1 and s["pooled"][1] < 1,
        })

    pd.concat(all_shifts, ignore_index=True).to_csv(
        ROBUST / "luna_calibration_shifts.csv", index=False)
    sl = pd.DataFrame(slope_rows)
    sl.to_csv(ROBUST / "luna_calibration_slopes.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== RQ2 CES-calibration slope: DeBERTa vs luna ===\n")
    for r in slope_rows:
        print(f"[{r['variant']:>20}]  slope OLS={r['slope_pooled_OLS']:.3f} "
              f"Deming={r['slope_pooled_Deming']:.3f}   "
              f"(L={r['slope_llama']:.2f} G={r['slope_gemma']:.2f} "
              f"Q={r['slope_qwen']:.2f} T={r['slope_gpt56terra']:.2f} "
              f"S={r['slope_sonnet5']:.2f})")
        print(f"{'':22}  republican={r['republican_shift']:+.3f}  "
              f"max|name|={r['max_abs_name_shift']:.3f}  slope<1={r['slope_below_1']}")
    print(f"\nWrote {ROBUST / 'luna_calibration_slopes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
