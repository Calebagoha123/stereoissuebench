#!/usr/bin/env python3
"""TOST equivalence tests for the implicit-cue (name / state) nulls.

A non-significant cue effect is not evidence of no effect. The two-one-sided-tests
(TOST) procedure (Lakens 2017, SPPS; Lakens, Scheel & Isager 2018) lets us make
the stronger, defensible claim: we can statistically reject name-cue effects as
large as X.

The setting hands us a principled, non-arbitrary smallest effect size of interest
(SESOI): the **real CES group difference** for the corresponding race x gender (or
state) group. A name-cue effect smaller than that real difference cannot be doing
the work calibrated personalisation would require; smaller than *half* of it is
behaviourally negligible relative to real between-group opinion gaps.

For each (model, implicit cue group) we run TOST against two symmetric bounds:
  - |CES group shift|           (full real difference)
  - 0.5 * |CES group shift|      (the stricter "half" bound)
Equivalence at level alpha is declared when the 90% CI of the model shift lies
entirely inside [-bound, +bound] (equivalently, both one-sided p < 0.05).

t-tests use df = (#issues - 1) = 18, the issue-clustering unit behind the
bootstrap SE. Reads results/robustness/model_shift_table.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, ROBUST

DF = 18  # 19 issues - 1 (the clustering unit)
ALPHA = 0.05

IMPLICIT = [
    ("implicit_demographic", "black_woman"),
    ("implicit_demographic", "black_man"),
    ("implicit_demographic", "white_woman"),
    ("implicit_demographic", "white_man"),
    ("implicit_political", "blue_state"),
    ("implicit_political", "red_state"),
    ("implicit_political", "swing_state"),
]


def tost(estimate, se, bound, df=DF, alpha=ALPHA):
    """Two one-sided t-tests against +/-bound. Returns dict with decision."""
    # upper: H0 estimate >= +bound ; reject if estimate significantly below +bound
    t_u = (estimate - bound) / se
    p_u = stats.t.cdf(t_u, df)          # lower-tail
    # lower: H0 estimate <= -bound ; reject if estimate significantly above -bound
    t_l = (estimate + bound) / se
    p_l = stats.t.sf(t_l, df)           # upper-tail
    p_tost = max(p_u, p_l)
    tcrit = stats.t.ppf(1 - alpha, df)
    ci90 = (estimate - tcrit * se, estimate + tcrit * se)
    equivalent = (p_tost < alpha) and (bound > 0)
    return {"p_tost": p_tost, "p_upper": p_u, "p_lower": p_l,
            "ci90_lo": ci90[0], "ci90_hi": ci90[1], "equivalent": equivalent}


def main():
    df = pd.read_csv(ROBUST / "model_shift_table.csv")
    recs = []
    for m in MODELS:
        for fam, grp in IMPLICIT:
            r = df[(df.model == m) & (df.cue_family == fam) & (df.cue_group == grp)]
            if r.empty:
                continue
            r = r.iloc[0]
            est, se = r["model_shift"], r["model_shift_se"]
            bound_full = abs(r["ces_shift_mean"])
            for tag, bound in [("full_ces", bound_full), ("half_ces", 0.5 * bound_full)]:
                res = tost(est, se, bound)
                recs.append({
                    "model": m, "cue_family": fam, "cue_group": grp,
                    "cue_display": r["cue_display"], "bound_type": tag,
                    "bound": bound, "model_shift": est, "model_shift_se": se,
                    **res,
                })
    out = pd.DataFrame(recs)
    out.to_csv(ROBUST / "tost_names.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== TOST equivalence: implicit cue effects vs real CES group difference ===")
    print("   'equivalent' = 90% CI within +/-bound  => reject effects as large as the bound\n")
    for m in MODELS:
        print(f"--- {MODEL_LABEL[m]} ---")
        sub = out[(out.model == m) & (out.bound_type == "full_ces")]
        for _, r in sub.iterrows():
            verdict = "EQUIV" if r["equivalent"] else "  -  "
            print(f"  {r['cue_display']:>22}  shift={r['model_shift']:+.3f} "
                  f"90%CI[{r['ci90_lo']:+.3f},{r['ci90_hi']:+.3f}]  "
                  f"bound=±{r['bound']:.3f}  p_TOST={r['p_tost']:.3f}  [{verdict}]")
        print()

    # Summary counts
    full = out[out.bound_type == "full_ces"]
    half = out[out.bound_type == "half_ces"]
    n = len(full)
    print(f"Equivalent to zero within the FULL real CES difference: "
          f"{full['equivalent'].sum()}/{n} (model x implicit-cue cells)")
    print(f"Equivalent within HALF the real CES difference:          "
          f"{half['equivalent'].sum()}/{n}")
    # Names only
    names_full = full[full.cue_family == "implicit_demographic"]
    print(f"\nName cues only — equivalent within full CES difference:  "
          f"{names_full['equivalent'].sum()}/{len(names_full)}")
    print(f"Wrote {ROBUST/'tost_names.csv'}")


if __name__ == "__main__":
    main()
