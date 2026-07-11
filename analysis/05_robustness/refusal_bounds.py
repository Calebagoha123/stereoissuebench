#!/usr/bin/env python3
"""Refusal robustness: Manski-style worst-case bounds on the cue effects.

Excluding refusals is selection on the outcome if refusal rates are not flat
across cues. Two moves:

1. Report the refusal rate per cue (is it flat?).
2. Treat refusals as unobserved stances and recompute each cue effect Delta_k
   under the worst-case assignments (Manski logic):
     - upper bound on Delta_k: cued refusals -> +1, baseline refusals -> -1
     - lower bound on Delta_k: cued refusals -> -1, baseline refusals -> +1
   (These span the simpler "all refusals coded 0 / coded to the opposing extreme"
   scenarios.) If the sign / conclusion survives both bounds, differential refusal
   cannot be driving it.

Refusals are only labelled in the judge-scored ``results/full/eval_<model>.csv``
run (the 3-repeat rerun did not retain response text locally), so the bounds are
computed there on the same {-1,0,+1} liberal-score scale. Refusal rates are <1.5%
and near-flat, so the bounds are tight by construction; this documents it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, CUE_ORDER, CUE_DISPLAY, ROBUST

FULL = Path("results/full")


def load_eval(model):
    d = pd.read_csv(FULL / f"eval_{model}.csv", low_memory=False,
                    usecols=["arm", "cue_family", "cue_group", "cue_condition",
                             "eval_label", "liberal_score"])
    d["refusal"] = (d["eval_label"] == "refusal")
    d["lib"] = pd.to_numeric(d["liberal_score"], errors="coerce")
    return d


def group_stats(sub):
    """n, refusals, sum of liberal score over non-refusals."""
    n = len(sub)
    r = int(sub["refusal"].sum())
    s = float(sub.loc[~sub["refusal"], "lib"].fillna(0).sum())
    return n, r, s


def main():
    recs = []
    for m in MODELS:
        d = load_eval(m)
        base = d[d["cue_family"] == "baseline"]
        nb, rb, sb = group_stats(base)
        base_obs = sb / (nb - rb) if nb > rb else np.nan
        for fam, grp in CUE_ORDER:
            cue = d[(d["cue_family"] == fam) & (d["cue_group"] == grp)]
            if cue.empty:
                continue
            # arm-B cues have no baseline within arm; use the shared baseline
            nc, rc, sc = group_stats(cue)
            cue_obs = sc / (nc - rc) if nc > rc else np.nan
            delta_obs = cue_obs - base_obs
            # worst-case means
            up = (sc + rc * 1) / nc - (sb + rb * -1) / nb   # cue refusals +1, base -1
            lo = (sc + rc * -1) / nc - (sb + rb * 1) / nb   # cue refusals -1, base +1
            # all-refusals-to-0 (both sides): refusals contribute 0
            zero = sc / nc - sb / nb
            recs.append({
                "model": m, "cue_family": fam, "cue_group": grp,
                "cue_display": CUE_DISPLAY[(fam, grp)],
                "refusal_rate_cue": rc / nc, "refusal_rate_base": rb / nb,
                "delta_observed": delta_obs, "delta_zero_coded": zero,
                "delta_manski_lo": lo, "delta_manski_hi": up,
                "sign_robust": np.sign(lo) == np.sign(up) and lo != 0,
            })
    out = pd.DataFrame(recs)
    out.to_csv(ROBUST / "refusal_bounds.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== Refusal robustness (Manski worst-case bounds on Delta_k) ===\n")
    print(f"Overall refusal rate range across cues: "
          f"{out['refusal_rate_cue'].min()*100:.2f}%–{out['refusal_rate_cue'].max()*100:.2f}%\n")
    for m in MODELS:
        s = out[out.model == m]
        print(f"--- {MODEL_LABEL[m]} ---")
        for _, r in s.iterrows():
            robust = "sign robust" if r["sign_robust"] else "  --  "
            print(f"  {r['cue_display']:>22}  ref={r['refusal_rate_cue']*100:4.1f}%  "
                  f"Δobs={r['delta_observed']:+.3f}  Manski[{r['delta_manski_lo']:+.3f},{r['delta_manski_hi']:+.3f}]  [{robust}]")
        print()
    # headline: do any large effects flip sign under worst case?
    big = out[out["delta_observed"].abs() > 0.05]
    flipped = big[~big["sign_robust"]]
    print(f"Cue effects with |Δ|>0.05: {len(big)}; of these, sign NOT robust to worst-case refusal: {len(flipped)}")
    if len(flipped):
        print(flipped[["model", "cue_display", "delta_observed", "delta_manski_lo", "delta_manski_hi"]].to_string(index=False))
    print(f"\nWrote {ROBUST/'refusal_bounds.csv'}")


if __name__ == "__main__":
    main()
