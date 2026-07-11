#!/usr/bin/env python3
"""DiD variance propagation: add the CES design-based sampling variance.

DiD_k = (model cued-minus-baseline shift) - (CES subgroup-minus-population shift).
The published bootstrap interval covers only the model term. The CES term is also
an estimate with design-based sampling variance; since the two are independent,

    Var(DiD_k) = Var_boot(model shift) + Var_design(CES shift).

We compute the CES design variance per issue by survey linearization and add it.
Writing the population shift as sub - pop = (1 - f)(sub - comp) with f the
subgroup's weight share and comp its complement, the per-issue design variance is

    v_k = (1 - f_k)^2 * [ Var_design(sub_k) + Var_design(comp_k) ]

with Var_design(mean) = sum_i w_i^2 (x_i - m)^2 / (sum_i w_i)^2 over the group's
opinionated respondents. Averaging equally over the K=19 designed issues,
Var_design(CES shift) = (1/K^2) * sum_k v_k. For small subgroups (Black men /
women) this term is non-negligible and visibly widens the DiD interval.

Reads results/robustness/model_shift_table.csv + the CES microdata. Writes an
augmented DiD table and a calibration figure with propagated x-errors.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, MODEL_COLOUR, ROBUST

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"
ISSUES = "data/input/issues_experiment.csv"
STATES = "data/input/states/state_bank.csv"
Z = 1.96


def leading_ints(cell):
    out = []
    for seg in str(cell).split(";"):
        m = re.match(r"\s*(\d+)", seg)
        if m:
            out.append(int(m.group(1)))
    return out


def wmean(x, w):
    ok = x.notna() & w.notna()
    return (x[ok] * w[ok]).sum() / w[ok].sum()


def design_var_mean(x, w):
    """Linearization design variance of a survey-weighted mean."""
    ok = x.notna() & w.notna()
    xi, wi = x[ok].to_numpy(), w[ok].to_numpy()
    if len(xi) < 2:
        return np.nan
    m = (wi * xi).sum() / wi.sum()
    return float((wi ** 2 * (xi - m) ** 2).sum() / (wi.sum() ** 2))


def ces_design_variance():
    """Per (cue_family, cue_group): design-based variance of the CES shift."""
    issues = pd.read_csv(ISSUES)
    issues = issues[issues["analysis_tier"] == "main"].copy()
    issue_vars = issues["ces_variable"].tolist()
    raw = pd.read_stata(CES_DTA, columns=["commonweight", "pid3", "race", "gender4"] + issue_vars,
                        convert_categoricals=False)
    st = pd.read_stata(CES_DTA, columns=["inputstate"], convert_categoricals=True)
    raw["state"] = st["inputstate"].astype(str).values
    w = raw["commonweight"].astype(float)

    lib = pd.DataFrame(index=raw.index)
    for _, row in issues.iterrows():
        v, sign = row["ces_variable"], int(row["liberal_sign"])
        sup = set(leading_ints(row["ces_support_code"]))
        opp = set(leading_ints(row["ces_oppose_code"]))
        val = pd.Series(np.nan, index=raw.index)
        val[raw[v].isin(sup)] = 1.0 * sign
        val[raw[v].isin(opp)] = -1.0 * sign
        lib[v] = val

    sb = pd.read_csv(STATES)
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

    recs = []
    for (fam, grp), mask in masks.items():
        vks = []
        for v in issue_vars:
            vals = lib[v]
            sub_ok = mask & vals.notna() & w.notna()
            comp_ok = (~mask) & vals.notna() & w.notna()
            if sub_ok.sum() < 2 or comp_ok.sum() < 2:
                continue
            f = w[sub_ok].sum() / (w[sub_ok].sum() + w[comp_ok].sum())
            vsub = design_var_mean(vals.where(mask), w.where(mask))
            vcomp = design_var_mean(vals.where(~mask), w.where(~mask))
            vks.append((1 - f) ** 2 * (vsub + vcomp))
        vks = np.array(vks)
        K = len(vks)
        var_design = float(vks.sum() / K ** 2) if K else np.nan
        rec = {"cue_family": fam, "cue_group": grp,
               "ces_design_var": var_design, "ces_design_se": np.sqrt(var_design)}
        recs.append(rec)
        if fam == "explicit_demographic":
            recs.append({"cue_family": "implicit_demographic", "cue_group": grp,
                         "ces_design_var": var_design, "ces_design_se": np.sqrt(var_design)})
    return pd.DataFrame(recs)


def main():
    mt = pd.read_csv(ROBUST / "model_shift_table.csv")
    cesv = ces_design_variance()
    df = mt.merge(cesv, on=["cue_family", "cue_group"], how="left")

    df["did"] = df["model_shift"] - df["ces_shift_mean"]
    # model-only interval (published) vs propagated
    df["se_model_only"] = df["model_shift_se"]
    df["se_did_propagated"] = np.sqrt(df["model_shift_var"] + df["ces_design_var"])
    df["did_lo"] = df["did"] - Z * df["se_did_propagated"]
    df["did_hi"] = df["did"] + Z * df["se_did_propagated"]
    df["ci_widening_pct"] = 100 * (df["se_did_propagated"] / df["se_model_only"] - 1)

    cols = ["model", "cue_family", "cue_group", "cue_display", "subgroup_n",
            "model_shift", "ces_shift_mean", "did",
            "ces_design_se", "se_model_only", "se_did_propagated",
            "did_lo", "did_hi", "ci_widening_pct"]
    out = df[cols]
    out.to_csv(ROBUST / "did_variance_propagated.csv", index=False)

    pd.set_option("display.width", 240)
    print("=== DiD with CES design variance propagated ===\n")
    # Show the widening, worst subgroups first
    w = (out.groupby(["cue_family", "cue_group", "cue_display", "subgroup_n"])
         .agg(ces_design_se=("ces_design_se", "first"),
              mean_widening=("ci_widening_pct", "mean")).reset_index()
         .sort_values("ces_design_se", ascending=False))
    print("CES design SE and mean DiD-interval widening by subgroup:")
    for _, r in w.iterrows():
        print(f"  {r['cue_display']:>22}  n={int(r['subgroup_n']):>5}  "
              f"CES design SE={r['ces_design_se']:.4f}  -> DiD SE +{r['mean_widening']:.1f}%")
    print(f"\nWrote {ROBUST/'did_variance_propagated.csv'}")
    make_figure(out)


def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    lim = 0.72
    xs = np.linspace(-lim, lim, 100)
    ax.plot(xs, xs, "--", color="#888", lw=1.1, label="calibration (y = x)")
    ax.axhline(0, color="#ccc", lw=0.7)
    ax.axvline(0, color="#ccc", lw=0.7)
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.errorbar(sub["ces_shift_mean"], sub["model_shift"],
                    yerr=Z * sub["se_model_only"],
                    xerr=Z * sub["ces_design_se"],
                    fmt="o", ms=6, color=MODEL_COLOUR[m], ecolor=MODEL_COLOUR[m],
                    elinewidth=0.8, capsize=2, alpha=0.85, label=MODEL_LABEL[m])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Real CES group shift (± design SE)")
    ax.set_ylabel("Model stance shift (± bootstrap SE)")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = Path("figures/robustness/did_calibration_propagated.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
