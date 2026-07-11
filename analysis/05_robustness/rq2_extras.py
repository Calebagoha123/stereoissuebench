#!/usr/bin/env python3
"""Secondary RQ2 robustness: multiplicity (BH-FDR), leave-one-issue-out slope,
and a distribution-free permutation test for the main cue effects.

1. **Multiplicity.** We frame the analysis as estimation (report effect sizes with
   CIs; interpret only large, replicated-across-models effects). As a companion for
   any per-cue significance claim we also control the false discovery rate with
   Benjamini-Hochberg across the family of 42 cue x model tests and report how many
   survive at FDR 0.05.

2. **Leave-one-issue-out.** Recompute the pooled calibration slope (OLS-through-
   origin and Deming) dropping each of the 19 issues in turn, to check whether a
   single high-salience issue (abortion / climate) drives it.

3. **Permutation test.** For each cue we shuffle the cue-vs-baseline label within
   issue x template cells to build a distribution-free null for Delta_k, as a
   companion to the issue-clustered bootstrap.

Reads the full_3x BERT data + the CES per-issue shifts. Writes three CSVs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import (MODELS, CUE_ORDER, CUE_DISPLAY, ROBUST, load_all)
from _regression import ols_through_origin, deming, x_var

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"
ISSUES = "data/input/issues_experiment.csv"
STATES = "data/input/states/state_bank.csv"
DF_ISS = 18


# ---------------------------------------------------------------- CES per issue
def leading_ints(cell):
    return [int(re.match(r"\s*(\d+)", s).group(1)) for s in str(cell).split(";")
            if re.match(r"\s*(\d+)", s)]


def ces_shift_per_issue():
    issues = pd.read_csv(ISSUES)
    issues = issues[issues["analysis_tier"] == "main"].copy()
    ivars = issues["ces_variable"].tolist()
    raw = pd.read_stata(CES_DTA, columns=["commonweight", "pid3", "race", "gender4"] + ivars,
                        convert_categoricals=False)
    st = pd.read_stata(CES_DTA, columns=["inputstate"], convert_categoricals=True)
    raw["state"] = st["inputstate"].astype(str).values
    w = raw["commonweight"].astype(float)
    lib = {}
    for _, row in issues.iterrows():
        v, sign = row["ces_variable"], int(row["liberal_sign"])
        sup, opp = set(leading_ints(row["ces_support_code"])), set(leading_ints(row["ces_oppose_code"]))
        val = pd.Series(np.nan, index=raw.index)
        val[raw[v].isin(sup)] = sign
        val[raw[v].isin(opp)] = -sign
        lib[v] = val
    sb = pd.read_csv(STATES); scat = dict(zip(sb["state"], sb["category"])); scat.setdefault("District of Columbia", "blue_state")
    cat = raw["state"].map(scat)
    masks = {
        ("explicit_political", "democrat"): raw["pid3"] == 1,
        ("explicit_political", "republican"): raw["pid3"] == 2,
        ("explicit_political", "independent"): raw["pid3"] == 3,
        ("explicit_demographic", "white_man"): (raw["race"] == 1) & (raw["gender4"] == 1),
        ("explicit_demographic", "white_woman"): (raw["race"] == 1) & (raw["gender4"] == 2),
        ("explicit_demographic", "black_man"): (raw["race"] == 2) & (raw["gender4"] == 1),
        ("explicit_demographic", "black_woman"): (raw["race"] == 2) & (raw["gender4"] == 2),
        ("implicit_political", "blue_state"): cat == "blue_state",
        ("implicit_political", "red_state"): cat == "red_state",
        ("implicit_political", "swing_state"): cat == "swing_state",
    }

    def wm(vals, mask):
        ok = vals.notna() & mask
        return (vals[ok] * w[ok]).sum() / w[ok].sum() if w[ok].sum() > 0 else np.nan

    recs = []
    for (fam, grp), mask in masks.items():
        for v in ivars:
            shift = wm(lib[v], mask) - wm(lib[v], pd.Series(True, index=raw.index))
            recs.append({"cue_family": fam, "cue_group": grp, "issue_id": v, "ces_shift": shift})
            if fam == "explicit_demographic":
                recs.append({"cue_family": "implicit_demographic", "cue_group": grp, "issue_id": v, "ces_shift": shift})
    return pd.DataFrame(recs)


# ------------------------------------------------- model per-issue aggregates
def model_issue_aggs(df):
    """Per (model, cue, issue): sum & count of y for the cue and its baseline.

    Arm-B cues get a template-matched baseline; Arm-A cues use the full baseline.
    """
    rows = []
    for model in MODELS:
        d = df[df.model == model]
        base = d[d.cue_family == "baseline"]
        base_g = base.groupby(["issue_id", "template_id"])["y"].agg(["sum", "count"])
        for fam, grp in CUE_ORDER:
            cue = d[(d.cue_family == fam) & (d.cue_group == grp)]
            if cue.empty:
                continue
            tmpl = set(cue["template_id"].unique()) if fam.startswith("implicit") else None
            for iss, g in cue.groupby("issue_id"):
                csum, cn = g["y"].sum(), len(g)
                if tmpl is None:
                    bsub = base[base.issue_id == iss]
                else:
                    bsub = base[(base.issue_id == iss) & (base.template_id.isin(tmpl))]
                rows.append({"model": model, "cue_family": fam, "cue_group": grp, "issue_id": iss,
                             "cue_sum": csum, "cue_n": cn,
                             "base_sum": bsub["y"].sum(), "base_n": len(bsub)})
    return pd.DataFrame(rows)


def shift_from_aggs(sub):
    return sub["cue_sum"].sum() / sub["cue_n"].sum() - sub["base_sum"].sum() / sub["base_n"].sum()


# ---------------------------------------------------------------- 1. multiplicity
def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals)
    order = np.argsort(p)
    n = len(p)
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = p[order] <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    reject = np.zeros(n, bool)
    if k > 0:
        reject[order[:k]] = True
    # BH-adjusted q-values
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        q[order[i]] = prev
    return reject, q


def multiplicity():
    mt = pd.read_csv(ROBUST / "model_shift_table.csv")
    t = mt["model_shift"] / mt["model_shift_se"]
    mt["p_two_sided"] = 2 * stats.t.sf(t.abs(), DF_ISS)
    reject, q = bh_fdr(mt["p_two_sided"].to_numpy(), 0.05)
    mt["bh_reject_fdr05"] = reject
    mt["bh_qvalue"] = q
    naive = (mt["p_two_sided"] < 0.05).sum()
    mt[["model", "cue_display", "model_shift", "model_shift_se", "p_two_sided",
        "bh_qvalue", "bh_reject_fdr05"]].to_csv(ROBUST / "multiplicity_bh.csv", index=False)
    print("=== Multiplicity (family of 42 cue x model tests) ===")
    print(f"  significant at naive p<0.05:      {naive}/42")
    print(f"  survive Benjamini-Hochberg FDR<5%: {reject.sum()}/42")
    print(f"  Wrote {ROBUST/'multiplicity_bh.csv'}\n")


# ---------------------------------------------------- 2. leave-one-issue-out slope
def loo_slope(aggs, ces_iss):
    mt = pd.read_csv(ROBUST / "model_shift_table.csv")
    issues = sorted(aggs["issue_id"].unique())
    ces_full = pd.read_csv(Path("results/full_3x/ces_estimates.csv"))
    recs = []
    for drop in [None] + issues:
        keep_aggs = aggs if drop is None else aggs[aggs.issue_id != drop]
        keep_ces = ces_iss if drop is None else ces_iss[ces_iss.issue_id != drop]
        # model shift per (model,cue)
        ms = (keep_aggs.groupby(["model", "cue_family", "cue_group"])
              .apply(shift_from_aggs, include_groups=False).rename("model_shift").reset_index())
        cs = (keep_ces.groupby(["cue_family", "cue_group"])["ces_shift"].mean()
              .rename("ces_shift_mean").reset_index())
        pts = ms.merge(cs, on=["cue_family", "cue_group"])
        # variances for Deming delta: reuse full-table vx/vy (approx, stable)
        pts = pts.merge(mt[["model", "cue_family", "cue_group", "model_shift_var",
                            "ces_shift_ci_low", "ces_shift_ci_high"]],
                        on=["model", "cue_family", "cue_group"], how="left")
        x, y = pts["ces_shift_mean"].to_numpy(), pts["model_shift"].to_numpy()
        b_o, _ = ols_through_origin(x, y)
        delta = pts["model_shift_var"].mean() / x_var(pts).mean()
        b_d, _ = deming(x, y, delta)
        recs.append({"dropped_issue": drop or "(none)", "ols_origin_slope": b_o, "deming_slope": b_d})
    out = pd.DataFrame(recs)
    out.to_csv(ROBUST / "leave_one_issue_out.csv", index=False)
    base = out[out.dropped_issue == "(none)"].iloc[0]
    rng = out[out.dropped_issue != "(none)"]
    print("=== Leave-one-issue-out pooled calibration slope ===")
    print(f"  full-sample:   OLS(origin)={base['ols_origin_slope']:.3f}  Deming={base['deming_slope']:.3f}")
    print(f"  LOO OLS range: [{rng['ols_origin_slope'].min():.3f}, {rng['ols_origin_slope'].max():.3f}]")
    print(f"  LOO Deming range:[{rng['deming_slope'].min():.3f}, {rng['deming_slope'].max():.3f}]")
    worst = rng.iloc[(rng["deming_slope"] - base["deming_slope"]).abs().argmax()]
    print(f"  most influential issue (Deming): {worst['dropped_issue']} -> {worst['deming_slope']:.3f}")
    print(f"  Wrote {ROBUST/'leave_one_issue_out.csv'}\n")


# ------------------------------------------------------- 3. permutation test
def permutation_test(df, n_perm=1000, seed=11):
    rng = np.random.default_rng(seed)
    recs = []
    for model in MODELS:
        d = df[df.model == model]
        base = d[d.cue_family == "baseline"]
        for fam, grp in CUE_ORDER:
            cue = d[(d.cue_family == fam) & (d.cue_group == grp)]
            if cue.empty:
                continue
            if fam.startswith("implicit"):
                tmpl = set(cue["template_id"].unique())
                b = base[base.template_id.isin(tmpl)]
            else:
                b = base
            pool = pd.concat([cue.assign(g=1), b.assign(g=0)], ignore_index=True)
            obs = pool.loc[pool.g == 1, "y"].mean() - pool.loc[pool.g == 0, "y"].mean()
            # stratify permutation by issue
            y = pool["y"].to_numpy()
            g = pool["g"].to_numpy()
            iss = pool["issue_id"].to_numpy()
            strata = {i: np.where(iss == i)[0] for i in np.unique(iss)}
            n1 = g.sum()
            null = np.empty(n_perm)
            for p in range(n_perm):
                gp = np.zeros_like(g)
                for idx in strata.values():
                    k = int(g[idx].sum())
                    if k:
                        chosen = rng.choice(idx, size=k, replace=False)
                        gp[chosen] = 1
                null[p] = y[gp == 1].mean() - y[gp == 0].mean()
            pval = (np.abs(null) >= abs(obs)).mean()
            recs.append({"model": model, "cue_display": CUE_DISPLAY[(fam, grp)],
                         "delta": obs, "perm_p": pval})
    out = pd.DataFrame(recs)
    out.to_csv(ROBUST / "permutation_test.csv", index=False)
    print("=== Permutation test (shuffle cue label within issue strata) ===")
    agree = ((out["perm_p"] < 0.05) == (out["delta"].abs() > 0.03)).mean()
    print(f"  {(out['perm_p']<0.05).sum()}/42 cue effects significant at perm p<0.05")
    print(f"  Wrote {ROBUST/'permutation_test.csv'}\n")


def main():
    df = load_all()
    multiplicity()
    ces_iss = ces_shift_per_issue()
    aggs = model_issue_aggs(df)
    loo_slope(aggs, ces_iss)
    permutation_test(df, n_perm=1000)


if __name__ == "__main__":
    main()
