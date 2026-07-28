#!/usr/bin/env python3
"""Cue x issue calibration (supplementary): does the cue-level calibration story
survive disaggregation to the issue level?

The headline RQ2 comparison aggregates each cue's model shift and CES shift over
the 19 issues before comparing them (70 cue x model cells). That aggregation could
hide issue-level miscalibration that cancels in the mean. Here we disaggregate:

  model side  per (model, cue, issue) shift = mean(y | cue, issue) -
              mean(y | baseline, issue), template-matched for Arm-B cues
              (same estimator as rq2_extras.model_issue_aggs);
  CES side    per (cue, issue) weighted subgroup - population difference
              (same recoding as rq2_extras.ces_shift_per_issue).

Fits, on the merged (model x cue x issue) frame (~1,330 rows):
  1. pooled OLS-through-origin + free-intercept OLS of model shift on CES shift,
     with two-way cluster-robust SEs (issue and cue), since rows repeat issues
     across cues and cues across issues;
  2. per cue-family through-origin slopes (issue-level analogue of tab:threshold);
  3. a crossed mixed model: model_shift ~ ces_shift with random intercepts for
     issue, cue, and model (variance components on a single dummy group), the
     hierarchical specification acknowledging the crossed issue/cue/model
     structure.

This is a robustness companion to the descriptive cue-level plots, not a
replacement. Uses existing generations and the CES file only; no new model runs.

Reads results/full_3x/luna_eval_*.csv + the CES .dta.
Writes results/robustness/cue_issue_calibration.csv (merged frame)
and results/robustness/cue_issue_slopes.csv (fits).
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "05_robustness"))
from _common import ROBUST, load_all  # noqa: E402
from _regression import ols_through_origin  # noqa: E402
from rq2_extras import ces_shift_per_issue, model_issue_aggs  # noqa: E402


def merged_frame() -> pd.DataFrame:
    df = load_all()
    aggs = model_issue_aggs(df)
    aggs["model_shift"] = aggs.cue_sum / aggs.cue_n - aggs.base_sum / aggs.base_n
    ces = ces_shift_per_issue().rename(columns={"ces_shift": "ces_shift_issue"})
    m = aggs.merge(ces, on=["cue_family", "cue_group", "issue_id"], how="inner")
    m["cue"] = m.cue_family + "__" + m.cue_group
    return m


def two_way_cluster_ols(m: pd.DataFrame) -> dict:
    """Free-intercept OLS with Cameron-Gelbach-Miller two-way clustered SEs."""
    import statsmodels.formula.api as smf

    def fit(groups):
        return smf.ols("model_shift ~ ces_shift_issue", data=m).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
    r_iss = fit(m["issue_id"])
    r_cue = fit(m["cue"])
    r_int = fit(m["issue_id"].astype(str) + "|" + m["cue"])
    # CGM: V = V_iss + V_cue - V_intersection
    V = r_iss.cov_params() + r_cue.cov_params() - r_int.cov_params()
    se = np.sqrt(np.diag(V))
    return {"slope": r_iss.params["ces_shift_issue"],
            "intercept": r_iss.params["Intercept"],
            "slope_se_2way": se[1], "intercept_se_2way": se[0]}


def crossed_mixed_model(m: pd.DataFrame) -> dict:
    """model_shift ~ ces_shift + (1|issue) + (1|cue) + (1|model), crossed via
    variance components on a single all-rows group."""
    import statsmodels.formula.api as smf

    d = m.copy()
    d["one"] = 1
    vc = {"issue": "0 + C(issue_id)", "cue": "0 + C(cue)", "model": "0 + C(model)"}
    md = smf.mixedlm("model_shift ~ ces_shift_issue", d, groups=d["one"],
                     re_formula="0", vc_formula=vc)
    r = md.fit(reml=True, method="lbfgs", maxiter=500)
    return {"slope": r.fe_params["ces_shift_issue"],
            "slope_se": r.bse_fe["ces_shift_issue"],
            "intercept": r.fe_params["Intercept"],
            "intercept_se": r.bse_fe["Intercept"],
            "vc_issue": float(r.vcomp[list(vc).index("issue")]),
            "vc_cue": float(r.vcomp[list(vc).index("cue")]),
            "vc_model": float(r.vcomp[list(vc).index("model")]),
            "resid_var": float(r.scale), "converged": bool(r.converged)}


def main() -> int:
    m = merged_frame()
    m.to_csv(ROBUST / "cue_issue_calibration.csv", index=False)

    rows = []
    x, y = m["ces_shift_issue"].to_numpy(), m["model_shift"].to_numpy()
    bo, se = ols_through_origin(x, y)
    r = np.corrcoef(x, y)[0, 1]
    rows.append({"scope": "pooled", "fit": "ols_origin", "n": len(m),
                 "slope": bo, "se": se, "r": r})

    cl = two_way_cluster_ols(m)
    rows.append({"scope": "pooled", "fit": "ols_free_2way_cluster", "n": len(m),
                 "slope": cl["slope"], "se": cl["slope_se_2way"],
                 "intercept": cl["intercept"], "intercept_se": cl["intercept_se_2way"]})

    mm = crossed_mixed_model(m)
    rows.append({"scope": "pooled", "fit": "mixed_crossed", "n": len(m),
                 "slope": mm["slope"], "se": mm["slope_se"],
                 "intercept": mm["intercept"], "intercept_se": mm["intercept_se"],
                 "vc_issue": mm["vc_issue"], "vc_cue": mm["vc_cue"],
                 "vc_model": mm["vc_model"], "resid_var": mm["resid_var"],
                 "converged": mm["converged"]})

    # no-party pooled + per cue family
    np_m = m[m.cue_family != "explicit_political"]
    bo_np, se_np = ols_through_origin(np_m["ces_shift_issue"].to_numpy(),
                                      np_m["model_shift"].to_numpy())
    rows.append({"scope": "pooled_no_party", "fit": "ols_origin", "n": len(np_m),
                 "slope": bo_np, "se": se_np,
                 "r": np.corrcoef(np_m["ces_shift_issue"], np_m["model_shift"])[0, 1]})
    for fam, g in m.groupby("cue_family"):
        b, s = ols_through_origin(g["ces_shift_issue"].to_numpy(),
                                  g["model_shift"].to_numpy())
        rows.append({"scope": fam, "fit": "ols_origin", "n": len(g), "slope": b,
                     "se": s, "r": np.corrcoef(g["ces_shift_issue"], g["model_shift"])[0, 1]})

    out = pd.DataFrame(rows)
    out.to_csv(ROBUST / "cue_issue_slopes.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== Cue x issue calibration (issue-level disaggregation) ===")
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
