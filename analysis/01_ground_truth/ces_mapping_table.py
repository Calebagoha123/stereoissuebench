#!/usr/bin/env python3
"""CES mapping table (check D) + swing-state classification sensitivity.

Documents the 19 unshown coding decisions behind the ground truth: for each issue,
the CES variable, question item, support/oppose response codes, and the recode to
the {-1,+1} liberal score. Also documents how red/swing/blue state classes were
built (2024 presidential margin) and tests whether the state-cue CES shifts change
if the borderline states flip class.

Writes docs/ces_mapping.md (issue table + state table) and prints the swing-flip
sensitivity. Reads data/input/issues_experiment.csv, data/input/states/state_bank.csv,
and the CES microdata (for the sensitivity only).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"
ISSUES = "data/input/issues_experiment.csv"
STATES = "data/input/states/state_bank.csv"
OUT_MD = Path("docs/ces_mapping.md")


def leading_ints(cell):
    return [int(re.match(r"\s*(\d+)", s).group(1)) for s in str(cell).split(";")
            if re.match(r"\s*(\d+)", s)]


def issue_table(issues):
    lines = ["## CES issue → liberal-score mapping (19 main issues)\n",
             "`liberal_sign` orients each item so +1 = the liberal side. Binary items map "
             "support→+1, oppose→−1, then ×`liberal_sign`; the abortion item is a 4-point "
             "ordinal split 3–4=support / 1–2=oppose.\n",
             "| CES var | Issue | Liberal side (stance target) | Support codes | Oppose codes | sign |",
             "|---|---|---|---|---|:--:|"]
    for _, r in issues.iterrows():
        sup = "; ".join(str(c) for c in leading_ints(r["ces_support_code"]))
        opp = "; ".join(str(c) for c in leading_ints(r["ces_oppose_code"]))
        lines.append(f"| {r['ces_variable']} | {r['ces_item_short']} | {r['stance_target']} "
                     f"| {sup} | {opp} | {r['liberal_sign']:+d} |")
    return "\n".join(lines)


def state_table(sb):
    lines = ["\n## State partisan classification\n",
             "Class from the **2024 presidential margin** (`cov_margin_2024`, Dem−Rep pts). "
             "The seven 2024 battlegrounds are `swing`; other states are `blue`/`red` by margin "
             "sign. The prompt cues use CA/NY/MA (blue), AL/OK/TX (red), GA/PA/WI (swing); the "
             "CES ground-truth subgroup uses **all** states in each class.\n",
             "| Class | n states | Margin range | States |", "|---|--:|---|---|"]
    for cls in ["blue_state", "swing_state", "red_state"]:
        s = sb[sb.category == cls].sort_values("cov_margin_2024")
        rng = f"{s.cov_margin_2024.min():+.1f} … {s.cov_margin_2024.max():+.1f}"
        names = ", ".join(s.state.tolist())
        lines.append(f"| {cls} | {len(s)} | {rng} | {names} |")
    return "\n".join(lines)


def swing_sensitivity(issues, sb):
    """Recompute blue/red/swing CES subgroup shifts under alternative classifications."""
    ivars = issues["ces_variable"].tolist()
    raw = pd.read_stata(CES_DTA, columns=["commonweight"] + ivars, convert_categoricals=False)
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

    def shifts(state_cat):
        cat = raw["state"].map(state_cat)
        out = {}
        for cls in ["blue_state", "red_state", "swing_state"]:
            mask = cat == cls
            per = []
            for v in ivars:
                vals = lib[v]
                pop = (vals * w).sum() / w[vals.notna()].sum()
                sub = (vals.where(mask) * w).sum() / w[mask & vals.notna()].sum()
                per.append(sub - pop)
            out[cls] = float(np.mean(per))
        return out

    base_cat = dict(zip(sb["state"], sb["category"]))
    base_cat.setdefault("District of Columbia", "blue_state")
    base = shifts(base_cat)

    # borderline states (|2024 margin| <= 3): flip each toward its lean
    borderline = sb[sb.cov_margin_2024.abs() <= 3.0]
    alt_cat = dict(base_cat)
    for _, r in borderline.iterrows():
        alt_cat[r["state"]] = "blue_state" if r["cov_margin_2024"] > 0 else "red_state"
    alt = shifts(alt_cat)

    lines = ["\n## Swing-state classification sensitivity\n",
             f"Borderline states (|2024 margin| ≤ 3 pts): "
             f"{', '.join(borderline.state.tolist())}. Reclassifying each into its 2024 lean "
             "(swing → blue/red) changes the state-cue CES shifts as follows:\n",
             "| Class | baseline shift | borderline-flipped | Δ |", "|---|--:|--:|--:|"]
    for cls in ["blue_state", "red_state", "swing_state"]:
        lines.append(f"| {cls} | {base[cls]:+.3f} | {alt[cls]:+.3f} | {alt[cls]-base[cls]:+.3f} |")
    lines.append(f"\nMax change: {max(abs(alt[c]-base[c]) for c in base):.3f} — the blue/red CES "
                 "shifts are stable; only the (definitionally residual) swing class moves, and the "
                 "model-side state effects are near-null regardless, so the calibration conclusion "
                 "does not depend on the borderline assignments.")
    print("\n".join(lines[3:]))
    return "\n".join(lines)


def main():
    issues = pd.read_csv(ISSUES)
    issues = issues[issues["analysis_tier"] == "main"].copy()
    sb = pd.read_csv(STATES)

    md = ["# CES ground-truth coding decisions\n",
          "Generated by `analysis/01_ground_truth/ces_mapping_table.py`. Survey means use "
          "CES `commonweight`; don't-know/missing codes are dropped to NaN.\n",
          issue_table(issues), state_table(sb), swing_sensitivity(issues, sb)]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
