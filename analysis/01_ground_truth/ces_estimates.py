#!/usr/bin/env python3
"""Weighted CES 2025 subgroup estimates for the calibration figure.

Reads the CES 2025 Common Content microdata (Stata .dta) and, for each of the 19
main issues, recodes each respondent's answer to a liberal score in {-1, +1}
(support of the liberal side = +1), using the support/oppose code mapping in
data/input/issues_experiment.csv. CES is forced-choice, so there is no 0 here.

For every cue subgroup we compute, per issue, the survey-weighted mean liberal
score for the subgroup and for the whole population, take the difference
(subgroup - population), then average those per-issue shifts across the 19 issues
(equal weight per issue, matching the issue-clustered model estimator). 95% CIs
are the normal approximation over the per-issue shifts.

Subgroup -> CES definition:
  explicit_political   democrat/republican/independent   pid3
  explicit_demographic white/black x man/woman           race x gender4
  implicit_demographic (name)   same race x gender4 subgroup as the label cue
  implicit_political   blue/red/swing state              inputstate -> state_bank

Output: one row per (cue_family, cue_group) with ces_score_mean, ces_shift_mean,
ces_population_mean, and shift CIs. The name (implicit_demographic) rows carry the
same CES numbers as the corresponding label (explicit_demographic) rows, since a
name signalling "Black woman" maps to the same real subgroup.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

Z = 1.96

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"

CUE_LABELS = {
    ("explicit_political", "democrat"): "I am a Democrat",
    ("explicit_political", "republican"): "I am a Republican",
    ("explicit_political", "independent"): "I am an Independent",
    ("explicit_demographic", "black_woman"): "I am a Black woman",
    ("explicit_demographic", "black_man"): "I am a Black man",
    ("explicit_demographic", "white_woman"): "I am a White woman",
    ("explicit_demographic", "white_man"): "I am a White man",
    ("implicit_political", "blue_state"): "Resident of blue state",
    ("implicit_political", "red_state"): "Resident of red state",
    ("implicit_political", "swing_state"): "Resident of swing state",
    ("implicit_demographic", "black_woman"): "Name: black female",
    ("implicit_demographic", "black_man"): "Name: black male",
    ("implicit_demographic", "white_woman"): "Name: white female",
    ("implicit_demographic", "white_man"): "Name: white male",
}


def leading_ints(cell: str) -> list[int]:
    """Extract the leading integer code of each ';'-separated option segment."""
    out = []
    for seg in str(cell).split(";"):
        m = re.match(r"\s*(\d+)", seg)
        if m:
            out.append(int(m.group(1)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ces", default=CES_DTA)
    p.add_argument("--issues", default="data/input/issues_experiment.csv")
    p.add_argument("--states", default="data/input/states/state_bank.csv")
    p.add_argument("--out", default="results/full_3x/ces_estimates.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    issues = pd.read_csv(args.issues)
    issues = issues[issues["analysis_tier"] == "main"].copy()
    issue_vars = issues["ces_variable"].tolist()
    print(f"{len(issue_vars)} main issues: {issue_vars}")

    need_raw = ["commonweight", "pid3", "race", "gender4"] + issue_vars
    raw = pd.read_stata(args.ces, columns=need_raw, convert_categoricals=False)
    # inputstate needs labels (FIPS -> state name) to join the state bank
    st = pd.read_stata(args.ces, columns=["inputstate"], convert_categoricals=True)
    raw["state"] = st["inputstate"].astype(str).values

    w = raw["commonweight"].astype(float)

    # Recode each issue to a per-respondent liberal score in {-1, +1}, else NaN.
    lib = pd.DataFrame(index=raw.index)
    for _, row in issues.iterrows():
        v = row["ces_variable"]
        sign = int(row["liberal_sign"])
        sup = set(leading_ints(row["ces_support_code"]))
        opp = set(leading_ints(row["ces_oppose_code"]))
        ans = raw[v]
        val = pd.Series(np.nan, index=raw.index)
        val[ans.isin(sup)] = 1.0 * sign
        val[ans.isin(opp)] = -1.0 * sign
        lib[v] = val

    # State bank -> red/blue/swing (DC not in bank; overwhelmingly Democratic).
    sb = pd.read_csv(args.states)
    state_cat = dict(zip(sb["state"], sb["category"]))
    state_cat.setdefault("District of Columbia", "blue_state")
    cat = raw["state"].map(state_cat)

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

    def wmean(vals: pd.Series, weights: pd.Series) -> float:
        ok = vals.notna()
        sw = weights[ok].sum()
        return float((vals[ok] * weights[ok]).sum() / sw) if sw > 0 else np.nan

    def estimate(mask: pd.Series) -> dict:
        sub_by_issue, pop_by_issue, shift_by_issue = [], [], []
        for v in issue_vars:
            vals = lib[v]
            pop = wmean(vals, w)
            sub = wmean(vals.where(mask), w.where(mask))
            if np.isnan(sub) or np.isnan(pop):
                continue
            sub_by_issue.append(sub)
            pop_by_issue.append(pop)
            shift_by_issue.append(sub - pop)
        shift = np.array(shift_by_issue)
        n = len(shift)
        se = Z * shift.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        m = float(shift.mean())
        return {
            "ces_score_mean": float(np.mean(sub_by_issue)),
            "ces_population_mean": float(np.mean(pop_by_issue)),
            "ces_shift_mean": m,
            "ces_shift_ci_low": m - se,
            "ces_shift_ci_high": m + se,
            "ces_n_issues": n,
            "subgroup_n": int(mask.sum()),
        }

    records = []
    for (fam, grp), mask in masks.items():
        est = estimate(mask)
        records.append({"cue_family": fam, "cue_group": grp,
                        "cue_label": CUE_LABELS[(fam, grp)], **est})
        # name cue mirrors the same real subgroup as the demographic label
        if fam == "explicit_demographic":
            records.append({"cue_family": "implicit_demographic", "cue_group": grp,
                            "cue_label": CUE_LABELS[("implicit_demographic", grp)], **est})

    out = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    print(out[["cue_family", "cue_group", "subgroup_n", "ces_score_mean",
               "ces_shift_mean", "ces_shift_ci_low", "ces_shift_ci_high"]].to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
