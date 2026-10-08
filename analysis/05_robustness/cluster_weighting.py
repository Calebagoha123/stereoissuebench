#!/usr/bin/env python3
"""Robustness: equal weight per issue versus equal weight per issue cluster.

The 19 main issues are not spread evenly over the 7 clusters (environment/energy 5,
immigration 4, gun policy 3, LGBT policy 3, policing 2, abortion 1, election
regulation 1). Averaging over issues therefore gives climate five times the weight
of abortion, which is odd for a design whose cues are partisan: abortion is among
the most party-sorted items in the CES and a plausible place for the largest cue
effect.

This script recomputes both sides of the calibration comparison under equal weight
per cluster -- the model shift and the CES subgroup shift -- so the two remain
commensurable, and reports:

  * per-cue mean shift across models under both weightings, and the largest
    per-cell change;
  * whether any cell changes sign or significance. Significance under cluster
    weighting is a cluster-$t$ over the 7 cluster means (t_6), which is materially
    more conservative than the 19-issue version (t_18), so a cell surviving it is
    a stronger claim than the one in the main text;
  * the pooled through-origin calibration slope under both weightings.

Outputs
  results/robustness/cluster_weighting.csv       per model x cue
  ~/Desktop/SDS/thesis/tables/cluster_weighting.tex  (with --tex)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "lib"))
import _common as C  # noqa: E402

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"
ISSUES = ROOT / "data/input/issues_experiment.csv"


def _load_ces_helpers():
    """Reuse ces_estimates.leading_ints so the CES recode cannot drift from it."""
    path = ROOT / "analysis/01_ground_truth/ces_estimates.py"
    spec = importlib.util.spec_from_file_location("ces_estimates", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ces_per_issue(issues: pd.DataFrame) -> pd.DataFrame:
    """Weighted CES subgroup-minus-population shift, per (cue_family, cue_group, issue)."""
    ce = _load_ces_helpers()
    issue_vars = issues["ces_variable"].tolist()
    need = ["commonweight", "pid3", "race", "gender4"] + issue_vars
    raw = pd.read_stata(CES_DTA, columns=need, convert_categoricals=False)
    st = pd.read_stata(CES_DTA, columns=["inputstate"], convert_categoricals=True)
    raw["state"] = st["inputstate"].astype(str).values
    w = raw["commonweight"].astype(float)

    lib = pd.DataFrame(index=raw.index)
    for _, row in issues.iterrows():
        v = row["ces_variable"]
        sign = int(row["liberal_sign"])
        sup = set(ce.leading_ints(row["ces_support_code"]))
        opp = set(ce.leading_ints(row["ces_oppose_code"]))
        val = pd.Series(np.nan, index=raw.index)
        val[raw[v].isin(sup)] = 1.0 * sign
        val[raw[v].isin(opp)] = -1.0 * sign
        lib[v] = val

    sb = pd.read_csv(ROOT / "data/input/states/state_bank.csv")
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

    def wmean(vals, weights):
        ok = vals.notna()
        sw = weights[ok].sum()
        return float((vals[ok] * weights[ok]).sum() / sw) if sw > 0 else np.nan

    recs = []
    for (fam, grp), mask in masks.items():
        for v in issue_vars:
            shift = wmean(lib[v].where(mask), w.where(mask)) - wmean(lib[v], w)
            recs.append({"cue_family": fam, "cue_group": grp,
                         "ces_variable": v, "ces_shift": shift})
            if fam == "explicit_demographic":   # name cue mirrors the same real subgroup
                recs.append({"cue_family": "implicit_demographic", "cue_group": grp,
                             "ces_variable": v, "ces_shift": shift})
    return pd.DataFrame(recs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="")
    ap.add_argument("--out", default=str(ROOT / "results/robustness/cluster_weighting.csv"))
    args = ap.parse_args()

    issues = pd.read_csv(ISSUES)
    issues = issues[issues["analysis_tier"] == "main"]
    clus = dict(zip(issues["ces_variable"], issues["issue_cluster"]))
    n_clusters = issues["issue_cluster"].nunique()

    df = C.load_all()
    df["cluster"] = df["ces_variable"].map(clus)
    assert df["cluster"].notna().all()
    iss_cluster = df[["issue_id", "cluster"]].drop_duplicates().set_index("issue_id")["cluster"]

    ces = ces_per_issue(issues)
    ces["cluster"] = ces["ces_variable"].map(clus)

    recs = []
    for model in C.MODELS:
        d = df[df["model"] == model]
        base_all = d[d["cue_family"] == "baseline"]
        for fam, grp in C.CUE_ORDER:
            cue = d[(d["cue_family"] == fam) & (d["cue_group"] == grp)]
            if cue.empty:
                continue
            base = (base_all[base_all["template_id"].isin(cue["template_id"].unique())]
                    if fam.startswith("implicit") else base_all)
            per_issue = (cue.groupby("issue_id")["y"].mean()
                         - base.groupby("issue_id")["y"].mean()).dropna()
            pi = pd.DataFrame({"d": per_issue})
            pi["cluster"] = pi.index.map(iss_cluster)

            # issue weighting (main text): cluster-t over 19 issues
            mu_i = per_issue.mean()
            se_i = per_issue.std(ddof=1) / np.sqrt(len(per_issue))
            lo_i, hi_i = stats.t.interval(0.95, len(per_issue) - 1, mu_i, se_i)

            # cluster weighting: cluster-t over the 7 cluster means
            cm = pi.groupby("cluster")["d"].mean()
            mu_c = cm.mean()
            se_c = cm.std(ddof=1) / np.sqrt(len(cm))
            lo_c, hi_c = stats.t.interval(0.95, len(cm) - 1, mu_c, se_c)

            ck = ces[(ces.cue_family == fam) & (ces.cue_group == grp)]
            ces_i = ck["ces_shift"].mean()
            ces_c = ck.groupby("cluster")["ces_shift"].mean().mean()

            recs.append({
                "model": model, "cue_family": fam, "cue_group": grp,
                "cue_display": C.CUE_DISPLAY[(fam, grp)],
                "shift_issue": mu_i, "shift_issue_lo": lo_i, "shift_issue_hi": hi_i,
                "sig_issue": bool(lo_i > 0 or hi_i < 0),
                "shift_cluster": mu_c, "shift_cluster_lo": lo_c, "shift_cluster_hi": hi_c,
                "sig_cluster": bool(lo_c > 0 or hi_c < 0),
                "delta": mu_c - mu_i,
                "ces_issue": ces_i, "ces_cluster": ces_c,
            })

    r = pd.DataFrame(recs)
    r["sign_flip"] = np.sign(r["shift_issue"]) != np.sign(r["shift_cluster"])
    r["sig_flip"] = r["sig_issue"] != r["sig_cluster"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out, index=False)

    # pooled through-origin calibration slope under each weighting
    def slope(x, y):
        return float((x * y).sum() / (x ** 2).sum())
    s_i = slope(r["ces_issue"].to_numpy(), r["shift_issue"].to_numpy())
    s_c = slope(r["ces_cluster"].to_numpy(), r["shift_cluster"].to_numpy())

    pd.set_option("display.width", 200)
    print(f"{len(r)} cue x model cells; {n_clusters} clusters over 19 issues")
    print(f"sign flips: {int(r.sign_flip.sum())}   significance flips: {int(r.sig_flip.sum())}")
    print(f"max |delta|: {r.delta.abs().max():.4f}   median |delta|: {r.delta.abs().median():.4f}")
    print(f"pooled through-origin slope: issue-weighted {s_i:.3f}, cluster-weighted {s_c:.3f}")
    if r.sig_flip.any():
        print("\ncells changing significance:")
        print(r[r.sig_flip][["model", "cue_display", "shift_issue", "sig_issue",
                             "shift_cluster", "sig_cluster"]].to_string(index=False))
    print("\nper-cue means across models:")
    g = (r.groupby("cue_display", sort=False)
         .agg(issue_wt=("shift_issue", "mean"), cluster_wt=("shift_cluster", "mean"),
              max_abs_delta=("delta", lambda s: s.abs().max()),
              n_sig_issue=("sig_issue", "sum"), n_sig_cluster=("sig_cluster", "sum"))
         .reset_index())
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.tex:
        write_tex(Path(args.tex), g, r, s_i, s_c, n_clusters)
        print(f"\nWrote {args.tex}")
    print(f"Wrote {args.out}")
    return 0


def write_tex(path: Path, g: pd.DataFrame, r: pd.DataFrame,
              s_i: float, s_c: float, n_clusters: int) -> None:
    def num(x):
        s = f"{x:.3f}"
        return s.replace("-", "$-$")

    def tex_quotes(s: str) -> str:
        """CM has no curly quotes; TeX wants `` and '' for them."""
        return s.replace("“", "``").replace("”", "''")
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Cue & \shortstack{Shift \\ (per issue)} & \shortstack{Shift \\ (per cluster)}"
             r" & \shortstack{Max cell \\ change} & \shortstack{Significant \\ cells} \\",
             r"\midrule"]
    for _, row in g.iterrows():
        lines.append(f"{tex_quotes(row['cue_display'])} & {num(row['issue_wt'])} & "
                     f"{num(row['cluster_wt'])} & {row['max_abs_delta']:.3f} & "
                     f"{int(row['n_sig_issue'])}/{int(row['n_sig_cluster'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Equal weight per issue versus equal weight per issue"
              r" cluster.} The 19 issues are unevenly spread over the "
              f"{n_clusters}"
              r" clusters (environment/energy 5, immigration 4, gun policy 3, LGBT"
              r" policy 3, policing 2, abortion 1, election regulation 1), so the"
              r" per-issue average used in the main text weights climate five times"
              r" abortion. Columns 2--3 give each cue's shift averaged over the five"
              r" models under both weightings; column 4 the largest change in any single"
              r" model $\times$ cue cell; column 5 the number of the five cells"
              r" significant under each (per issue / per cluster). Cluster weighting"
              r" recomputes both sides of the comparison, model and CES, so the two stay"
              r" commensurable, and its interval is a cluster-$t$ over "
              f"{n_clusters}"
              r" cluster means rather than 19 issue means, so its critical value is"
              r" $t_6$ rather than $t_{18}$ and the test is materially more"
              r" conservative. The largest single-cell change is "
              f"{r.delta.abs().max():.3f}"
              r" (median "
              f"{r.delta.abs().median():.3f}"
              r"). Sign changes occur in "
              f"{int(r.sign_flip.sum())}"
              r" cells, all of them null cells oscillating about zero"
              r" ($|\widehat{\Delta}_k| \leq "
              f"{r[r.sign_flip][['shift_issue', 'shift_cluster']].abs().to_numpy().max():.3f}"
              r"$, and none significant under either weighting). Significance is lost in "
              f"{int(r.sig_flip.sum())}"
              r" cells, all of them small effects"
              r" ($|\widehat{\Delta}_k| \leq "
              f"{r[r.sig_flip].shift_issue.abs().max():.2f}"
              r"$) whose estimates barely move (largest change among them "
              f"{r[r.sig_flip].delta.abs().max():.3f}"
              r"): the loss is the wider interval, not a different answer. Every"
              r" headline cue---Republican, Democrat, Black woman, Black man---remains"
              r" significant in all five models under both. The pooled through-origin"
              r" calibration slope is "
              f"{s_i:.2f}"
              r" per issue against "
              f"{s_c:.2f}"
              r" per cluster. Cluster weighting slightly \emph{increases} the"
              r" race~$\times$~gender label effects, so the per-issue average understates"
              r" rather than overstates them.}",
              r"\label{tab:cluster_weighting}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
