#!/usr/bin/env python3
"""
Assemble the experiment results into a handful of tidy, self-describing CSVs
suitable for feeding to an LLM for pattern-finding / narrative.

Run from repo root:  python3 analysis/build_consolidated.py
Outputs -> results/consolidated/  (+ DATA_DICTIONARY.md)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ROB = RES / "robustness"
PROBE = RES / "probe_internal"
OUT = RES / "consolidated"
OUT.mkdir(exist_ok=True)


def rd(p):
    return pd.read_csv(p)


def parse_ci(s):
    """Pull (lo, hi) floats out of the messy '(np.float64(x) np.float64(y))' strings."""
    if pd.isna(s):
        return (np.nan, np.nan)
    nums = re.findall(r"-?\d+\.\d+(?:e-?\d+)?", str(s))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return (np.nan, np.nan)


MODEL_NORM = {
    "Qwen-3.6-27B": "qwen", "Gemma-3-12B": "gemma", "Llama-3.1-8B": "llama",
    "qwen": "qwen", "gemma": "gemma", "llama": "llama",
}

# ---------------------------------------------------------------------------
# 01  MASTER CUE-EFFECT TABLE  (grain: model x cue ; 42 rows)
#     RQ1 effect + RQ2 DiD-vs-CES + every per-cue robustness statistic
# ---------------------------------------------------------------------------
mst = rd(ROB / "model_shift_table.csv")
did = rd(ROB / "did_variance_propagated.csv")
tost = rd(ROB / "tost_names.csv")
ref = rd(ROB / "refusal_bounds.csv")
comp = rd(ROB / "composition_summary.csv")
mult = rd(ROB / "multiplicity_bh.csv")
perm = rd(ROB / "permutation_test.csv")
clmm = rd(ROB / "clmm_coefs.csv")

key = ["model", "cue_family", "cue_group"]

m = mst[[
    "model", "cue_family", "cue_group", "cue_display", "cue_label",
    "n_cue", "n_base", "subgroup_n",
    "model_shift", "model_shift_se", "model_shift_lo", "model_shift_hi",
    "ces_shift_mean", "ces_shift_ci_low", "ces_shift_ci_high",
]].copy()

m = m.merge(did[key + ["did", "did_lo", "did_hi", "se_did_propagated", "ci_widening_pct"]],
            on=key, how="left")
# TOST has two rows per cue (equivalence margin = full_ces vs half_ces); pivot to columns
tost_p = tost.pivot_table(index=key, columns="bound_type", values="p_tost")
tost_p.columns = [f"tost_p_{c}" for c in tost_p.columns]
tost_e = tost.pivot_table(index=key, columns="bound_type", values="equivalent")
tost_e.columns = [f"tost_equivalent_{c}" for c in tost_e.columns]
m = m.merge(tost_p.reset_index(), on=key, how="left").merge(tost_e.reset_index(), on=key, how="left")
m = m.merge(ref[key + ["refusal_rate_cue", "refusal_rate_base",
                       "delta_manski_lo", "delta_manski_hi", "sign_robust"]], on=key, how="left")
m = m.merge(comp[key + ["collapse_index", "collapse_index_se",
                        "frac_issues_more_extreme", "mean_model_extremity",
                        "mean_ces_extremity", "mean_neutral_rate"]], on=key, how="left")

# multiplicity / permutation key on (model, cue_display)
m = m.merge(mult[["model", "cue_display", "p_two_sided", "bh_qvalue", "bh_reject_fdr05"]],
            on=["model", "cue_display"], how="left")
m = m.merge(perm[["model", "cue_display", "perm_p"]], on=["model", "cue_display"], how="left")

# clmm keys on (model, "family__group")
clmm = clmm.rename(columns={"cue": "cue_fg"})
clmm[["cue_family", "cue_group"]] = clmm["cue_fg"].str.split("__", expand=True)
m = m.merge(clmm[key + ["clmm_logodds", "clmm_se", "clmm_p"]], on=key, how="left")

# derived descriptor columns
m.insert(1, "directness", m["cue_family"].str.split("_").str[0])          # explicit / implicit
m.insert(2, "domain", m["cue_family"].str.split("_").str[1])              # political / demographic
# per-cue calibration ratio; undefined where the CES shift is ~0 (e.g. Independent)
m["signed_calibration_ratio"] = np.where(
    m["ces_shift_mean"].abs() < 0.02, np.nan, m["model_shift"] / m["ces_shift_mean"])
m = m.sort_values(["model", "cue_family", "cue_group"]).reset_index(drop=True)
m.to_csv(OUT / "01_master_cue_effects.csv", index=False)
print(f"01_master_cue_effects.csv           {m.shape[0]:>5} rows x {m.shape[1]} cols")

# ---------------------------------------------------------------------------
# 02  CALIBRATION SLOPES  (RQ2 headline + sensitivity, one tidy table)
# ---------------------------------------------------------------------------
rq2 = rd(ROB / "rq2_regression.csv")
lo, hi = zip(*rq2["ols_origin_ci"].map(parse_ci))
rq2["ols_origin_ci_lo"], rq2["ols_origin_ci_hi"] = lo, hi
rows = []
for _, r in rq2.iterrows():
    rows.append(dict(
        block="headline", specification=f"scope={r['scope']}", n=r["n"],
        slope_ols_origin=r["ols_origin_slope"], slope_ols_origin_se=r["ols_origin_slope_se"],
        slope_ols_origin_lo=r["ols_origin_ci_lo"], slope_ols_origin_hi=r["ols_origin_ci_hi"],
        slope_deming=r["deming_slope"], slope_free=r["ols_free_slope"],
        free_intercept=r["ols_free_intercept"], free_intercept_p=r["ols_free_intercept_p"],
        slope_llama=np.nan, slope_gemma=np.nan, slope_qwen=np.nan, note="",
    ))

ss = rd(ROB / "stance_sensitivity.csv")
for _, r in ss.iterrows():
    rows.append(dict(
        block="sensitivity_threshold", specification=f"variant={r['variant']}", n=np.nan,
        slope_ols_origin=r["slope_pooled_OLSorigin"], slope_ols_origin_se=np.nan,
        slope_ols_origin_lo=np.nan, slope_ols_origin_hi=np.nan,
        slope_deming=r["slope_pooled_Deming"], slope_free=np.nan,
        free_intercept=np.nan, free_intercept_p=np.nan,
        slope_llama=r["slope_llama"], slope_gemma=r["slope_gemma"], slope_qwen=r["slope_qwen"],
        note=f"republican_strongest={r['republican_is_strongest']}; slope_below_1={r['slope_below_1']}",
    ))

loi = rd(ROB / "leave_one_issue_out.csv")
for _, r in loi.iterrows():
    rows.append(dict(
        block="sensitivity_leave_one_issue", specification=f"drop={r['dropped_issue']}", n=np.nan,
        slope_ols_origin=r["ols_origin_slope"], slope_ols_origin_se=np.nan,
        slope_ols_origin_lo=np.nan, slope_ols_origin_hi=np.nan,
        slope_deming=r["deming_slope"], slope_free=np.nan,
        free_intercept=np.nan, free_intercept_p=np.nan,
        slope_llama=np.nan, slope_gemma=np.nan, slope_qwen=np.nan, note="",
    ))
cal = pd.DataFrame(rows)
cal.to_csv(OUT / "02_calibration_slopes.csv", index=False)
print(f"02_calibration_slopes.csv           {cal.shape[0]:>5} rows x {cal.shape[1]} cols")

# ---------------------------------------------------------------------------
# 03  CES GROUND TRUTH  (per issue ; the RQ2 x-axis context)
# ---------------------------------------------------------------------------
iss = rd(RES / "full_3x" / "ces_descriptives_issues.csv")
iss.to_csv(OUT / "03_ces_ground_truth_by_issue.csv", index=False)
print(f"03_ces_ground_truth_by_issue.csv    {iss.shape[0]:>5} rows x {iss.shape[1]} cols")

# ---------------------------------------------------------------------------
# 04  PROBE / MECHANISM SUMMARY  (grain: model ; RQ3 headline)
# ---------------------------------------------------------------------------
lad = rd(PROBE / "ladder_summary.csv")
lad["model"] = lad["model"].map(MODEL_NORM)
pc = rd(PROBE / "probe_correlation_ci.csv")
pc["link_key"] = pc["link"].str.split().str[0]          # A2 / B3
piv = pc.pivot_table(index="model", columns="link_key",
                     values=["r", "ci_lo", "ci_hi", "perm_p"])
piv.columns = [f"{a.lower()}_{b}" for a, b in piv.columns]  # e.g. r_A2
piv = piv.reset_index()
probe = lad.merge(piv, on="model", how="left")
probe.to_csv(OUT / "04_probe_summary_by_model.csv", index=False)
print(f"04_probe_summary_by_model.csv       {probe.shape[0]:>5} rows x {probe.shape[1]} cols")

# ---------------------------------------------------------------------------
# 05  INTERNAL DECODABILITY BY LAYER  (RQ3 detail; stacked across models)
# ---------------------------------------------------------------------------
dec = []
for mdl in ["llama", "gemma", "qwen"]:
    d = rd(PROBE / f"{mdl}_decodability_by_layer.csv")
    d.insert(0, "model", mdl)
    dec.append(d)
dec = pd.concat(dec, ignore_index=True)
dec.to_csv(OUT / "05_probe_decodability_by_layer.csv", index=False)
print(f"05_probe_decodability_by_layer.csv  {dec.shape[0]:>5} rows x {dec.shape[1]} cols")

# ---------------------------------------------------------------------------
# 06  NAME LEGIBILITY  (behavioural name->identity inference; RQ3 mechanism)
# ---------------------------------------------------------------------------
leg = rd(RES / "cue_probe" / "legibility_by_subgroup.csv")
leg.to_csv(OUT / "06_name_legibility_by_subgroup.csv", index=False)
print(f"06_name_legibility_by_subgroup.csv  {leg.shape[0]:>5} rows x {leg.shape[1]} cols")

# ---------------------------------------------------------------------------
# 07  PCT ROBUSTNESS ARM  (Political Compass Test cue effects)
# ---------------------------------------------------------------------------
pct = rd(RES / "pct_names_full" / "pct_cue_effects.csv")
pct.to_csv(OUT / "07_pct_cue_effects.csv", index=False)
print(f"07_pct_cue_effects.csv              {pct.shape[0]:>5} rows x {pct.shape[1]} cols")

# ---------------------------------------------------------------------------
# 08  ISSUE-LEVEL DETAIL  (grain: model x cue x issue ; 855 rows). RQ1/RQ2 substrate.
#     The un-aggregated view: how each cue lands on each of the 19 issues.
# ---------------------------------------------------------------------------
cpi = rd(ROB / "composition_per_issue.csv")
cpi.insert(1, "directness", cpi["cue_family"].str.split("_").str[0])
cpi.insert(2, "domain", cpi["cue_family"].str.split("_").str[1])
# readable issue label
cpi = cpi.merge(iss[["ces_variable", "issue", "pop_liberal_pct", "dem_rep_gap"]],
                on="ces_variable", how="left")
# per-issue written-stance shift vs this model's baseline on the same issue
base = (cpi[cpi.cue_family == "baseline"][["model", "ces_variable", "model_lib_share"]]
        .rename(columns={"model_lib_share": "baseline_lib_share"}))
cpi = cpi.merge(base, on=["model", "ces_variable"], how="left")
cpi["issue_model_shift"] = cpi["model_lib_share"] - cpi["baseline_lib_share"]
cpi = cpi.sort_values(["model", "cue_family", "cue_group", "ces_variable"]).reset_index(drop=True)
cpi.to_csv(OUT / "08_issue_level_detail.csv", index=False)
print(f"08_issue_level_detail.csv          {cpi.shape[0]:>5} rows x {cpi.shape[1]} cols")

# ---------------------------------------------------------------------------
# 09  NAME/STATE INSTANCE DETAIL  (grain: model x cue x instance ; 1836 rows).
#     Per-name (562) and per-state effects — the least-aggregated implicit view.
# ---------------------------------------------------------------------------
ins = rd(ROB / "instance_effects.csv")
ins.insert(1, "directness", ins["cue_family"].str.split("_").str[0])
ins = ins.sort_values(["model", "cue_family", "cue_group", "instance"]).reset_index(drop=True)
ins.to_csv(OUT / "09_instance_effects.csv", index=False)
print(f"09_instance_effects.csv            {ins.shape[0]:>5} rows x {ins.shape[1]} cols")

print(f"\nWrote {len(list(OUT.glob('*.csv')))} CSVs to {OUT.relative_to(ROOT)}/")
