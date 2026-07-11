#!/usr/bin/env python3
"""Descriptives of the CES 2025 Common Content for the paper's data section.

Two tables, both restricted to the quantities the study actually uses as the
real-world ground truth for the cue subgroups:

  Table 1  Sample composition of the ground-truth subgroups (party, race,
           gender, race x gender, and state partisan class), as unweighted n
           and survey-weighted share. This documents who the CES targets are
           that each cue is meant to stand in for.
  Table 2  The 19 main issues: population survey-weighted liberal share and the
           real Democrat-Republican gap (the partisan polarization the model is
           later asked to reproduce). This motivates the calibration figure's
           x-axis.

Recode of each issue to a per-respondent liberal score in {-1, +1} is identical
to analysis/ces_estimates.py (forced choice, so no 0). Writes a markdown block
and two CSVs to results/full_3x/.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"


def leading_ints(cell: str) -> list[int]:
    out = []
    for seg in str(cell).split(";"):
        m = re.match(r"\s*(\d+)", seg)
        if m:
            out.append(int(m.group(1)))
    return out


def wshare(mask: pd.Series, w: pd.Series, valid: pd.Series) -> float:
    """Survey-weighted share of `mask` among `valid` respondents (percent)."""
    denom = w[valid].sum()
    return 100.0 * w[mask & valid].sum() / denom if denom > 0 else np.nan


def wmean(vals: pd.Series, w: pd.Series) -> float:
    ok = vals.notna()
    sw = w[ok].sum()
    return float((vals[ok] * w[ok]).sum() / sw) if sw > 0 else np.nan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ces", default=CES_DTA)
    p.add_argument("--issues", default="data/input/issues_experiment.csv")
    p.add_argument("--states", default="data/input/states/state_bank.csv")
    p.add_argument("--out-dir", default="results/full_3x")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    issues = pd.read_csv(args.issues)
    issues = issues[issues["analysis_tier"] == "main"].copy()
    issue_vars = issues["ces_variable"].tolist()

    cols = ["commonweight", "pid3", "race", "gender4"] + issue_vars
    raw = pd.read_stata(args.ces, columns=cols, convert_categoricals=False)
    st = pd.read_stata(args.ces, columns=["inputstate"], convert_categoricals=True)
    raw["state"] = st["inputstate"].astype(str).values
    w = raw["commonweight"].astype(float)
    n_total = len(raw)
    w_valid = w.notna() & (w > 0)

    # --- Table 1: subgroup composition --------------------------------------
    # (block label, category label, boolean mask, "valid universe" mask)
    all_ok = pd.Series(True, index=raw.index)
    party = {1: "Democrat", 2: "Republican", 3: "Independent", 4: "Other", 5: "Not sure"}
    rows = []

    def add(block, label, mask, universe=all_ok):
        rows.append({"block": block, "category": label,
                     "n_unweighted": int((mask & universe & w_valid).sum()),
                     "weighted_pct": round(wshare(mask, w, universe & w_valid), 1)})

    for code, lab in party.items():
        add("Party (pid3)", lab, raw["pid3"] == code)
    for code, lab in {1: "White", 2: "Black"}.items():
        add("Race (race)", lab, raw["race"] == code)
    add("Race (race)", "Other/multiple", ~raw["race"].isin([1, 2]) & raw["race"].notna())
    for code, lab in {1: "Man", 2: "Woman"}.items():
        add("Gender (gender4)", lab, raw["gender4"] == code)
    add("Gender (gender4)", "Non-binary/other", ~raw["gender4"].isin([1, 2]) & raw["gender4"].notna())

    # race x gender cells actually used as cue targets (universe = white/black x man/woman)
    rg_universe = raw["race"].isin([1, 2]) & raw["gender4"].isin([1, 2])
    for (rc, rl) in [(1, "White"), (2, "Black")]:
        for (gc, gl) in [(1, "man"), (2, "woman")]:
            add("Race x Gender (used)", f"{rl} {gl}",
                (raw["race"] == rc) & (raw["gender4"] == gc), rg_universe)

    # state partisan class via the study's own state bank
    sb = pd.read_csv(args.states)
    state_cat = dict(zip(sb["state"], sb["category"]))
    state_cat.setdefault("District of Columbia", "blue_state")
    cat = raw["state"].map(state_cat)
    st_universe = cat.notna()
    for c, lab in [("blue_state", "Blue state"), ("swing_state", "Swing state"),
                   ("red_state", "Red state")]:
        add("State partisan class", lab, cat == c, st_universe)

    t1 = pd.DataFrame(rows)
    t1.to_csv(out_dir / "ces_descriptives_composition.csv", index=False)

    # --- Table 2: per-issue population liberal share + Dem-Rep gap -----------
    lib = pd.DataFrame(index=raw.index)
    for _, r in issues.iterrows():
        v = r["ces_variable"]
        sign = int(r["liberal_sign"])
        sup = set(leading_ints(r["ces_support_code"]))
        opp = set(leading_ints(r["ces_oppose_code"]))
        ans = raw[v]
        val = pd.Series(np.nan, index=raw.index)
        val[ans.isin(sup)] = 1.0 * sign
        val[ans.isin(opp)] = -1.0 * sign
        lib[v] = val

    dem = raw["pid3"] == 1
    rep = raw["pid3"] == 2
    irows = []
    for _, r in issues.iterrows():
        v = r["ces_variable"]
        vals = lib[v]
        pop = wmean(vals, w)                       # mean on {-1,+1}
        dgap = wmean(vals.where(dem), w.where(dem)) - wmean(vals.where(rep), w.where(rep))
        irows.append({
            "ces_variable": v,
            "issue": r["ces_item_short"],
            "pop_liberal_pct": round(100.0 * (pop + 1) / 2, 1),  # share on liberal side
            "dem_rep_gap": round(dgap, 3),
            "n": int(vals.notna().sum()),
        })
    t2 = pd.DataFrame(irows).sort_values("dem_rep_gap", ascending=False)
    t2.to_csv(out_dir / "ces_descriptives_issues.csv", index=False)

    # --- Markdown for the paper ---------------------------------------------
    lines = []
    lines.append("## CES 2025 Common Content — descriptives\n")
    lines.append(f"Total respondents: **{n_total:,}** (unweighted); "
                 f"all shares below use the CES common-content survey weight "
                 f"(`commonweight`).\n")
    lines.append("### Table 1. Ground-truth subgroup composition\n")
    lines.append("| Variable | Category | n (unwt.) | Weighted % |")
    lines.append("|---|---|---:|---:|")
    prev = None
    for _, r in t1.iterrows():
        blk = r["block"] if r["block"] != prev else ""
        prev = r["block"]
        lines.append(f"| {blk} | {r['category']} | {r['n_unweighted']:,} | {r['weighted_pct']:.1f} |")
    lines.append("\n*Race × Gender shares are within the white/black × man/woman universe; "
                 "state shares are within states mapped by the study's state bank.*\n")

    lines.append("### Table 2. The 19 main issues, real opinion structure\n")
    lines.append("Population liberal share and the real Democrat−Republican gap on the "
                 "signed liberal axis ({−1, +1}), sorted by partisan gap.\n")
    lines.append("| Issue | Pop. % liberal | Dem−Rep gap | n |")
    lines.append("|---|---:|---:|---:|")
    for _, r in t2.iterrows():
        lines.append(f"| {r['issue']} | {r['pop_liberal_pct']:.1f} | {r['dem_rep_gap']:+.3f} | {r['n']:,} |")
    lines.append(f"\nMean Dem−Rep gap across issues: **{t2['dem_rep_gap'].mean():+.3f}** "
                 f"(min {t2['dem_rep_gap'].min():+.3f}, max {t2['dem_rep_gap'].max():+.3f}).\n")

    md = "\n".join(lines)
    (out_dir / "ces_descriptives.md").write_text(md)
    print(md)
    print(f"\nWrote ces_descriptives.md + two CSVs to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
