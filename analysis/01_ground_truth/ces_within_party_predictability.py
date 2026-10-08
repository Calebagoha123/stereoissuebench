#!/usr/bin/env python3
"""How much of the *within-party* disagreement in CES is predictable from covariates?

The party-levels figure contrasts a degenerate model stance (the assistant writes
one side in effectively every response under "I am a Republican") with a CES party
share that is not degenerate (e.g. 62/38). The obvious objection is that the 38%
are not arbitrary: they might be the young, college-educated, non-religious or
coastal Republicans, so a responder given a fuller user profile could place them.

This script tests that. Within each party (pid3), for each of the 19 main issues,
it asks how well a respondent's side can be predicted from progressively larger
covariate sets, using 5-fold cross-validated weighted logistic regression:

  M0  intercept only            -- the party label alone (= the model's information)
  M1  + age, education, region, religiosity   -- the covariates named in the text
  M2  + gender, race, income, urbanicity, marital status, news interest
  M3  + pid7, ideo5             -- political self-placement the party cue never supplies

The headline quantity is the mismatch rate: the weighted share of that party's
respondents who receive writing on the wrong side of their own position.

  baseline mismatch  min(share, 1 - share)   -- one stance for everyone (the model)
  model mismatch     weighted CV error rate  -- a responder that conditions on X

If M1/M2 barely move the mismatch rate below baseline, the within-party split is
irreducible from observables: no amount of extra identity disclosure would let a
responder know which side of the proposition a given Republican falls on.

A second table ranks single covariate blocks by cross-validated AUC, which is the
"which variables predict support" question asked in a form that survives a binary,
weighted, ceiling-prone outcome (as opposed to comparing raw coefficients).

Outputs
  results/full_3x/ces_within_party_predictability.csv        per party x issue x model
  results/full_3x/ces_within_party_blocks.csv                per party x issue x block
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CES_DTA = "/Users/calebagoha/Desktop/SDS/thesis-experiments/CES/CES25_Common.dta"
SURVEY_YEAR = 2025
N_FOLDS = 5
SEED = 20250803

PARTIES = {"republican": 2, "democrat": 1}

# covariate blocks -> (categorical vars, continuous vars)
BLOCKS: dict[str, tuple[list[str], list[str]]] = {
    "age": ([], ["age", "age_sq"]),
    "education": (["educ"], []),
    "region": (["region"], []),
    "religiosity": (["pew_religimp", "pew_churatd", "pew_bornagain"], []),
    "gender_race": (["gender4", "race5"], []),
    "income": (["faminc_cat"], []),
    "urbanicity": (["urbancity"], []),
    "marital": (["marstat"], []),
    "news_interest": (["newsint"], []),
    "pid7": (["pid7"], []),
    "ideology": (["ideo5"], []),
}

MODELS: dict[str, list[str]] = {
    "M0_party_only": [],
    "M1_named": ["age", "education", "region", "religiosity"],
    "M2_all_demographics": [
        "age", "education", "region", "religiosity",
        "gender_race", "income", "urbanicity", "marital", "news_interest",
    ],
    "M3_plus_political_identity": [
        "age", "education", "region", "religiosity",
        "gender_race", "income", "urbanicity", "marital", "news_interest",
        "pid7", "ideology",
    ],
}

RAW_VARS = [
    "commonweight", "pid3", "pid7", "ideo5", "birthyr", "educ", "region",
    "pew_religimp", "pew_churatd", "pew_bornagain", "gender4", "race",
    "faminc_new", "urbancity", "marstat", "newsint",
]


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
    p.add_argument("--out", default="results/full_3x/ces_within_party_predictability.csv")
    p.add_argument("--out-blocks", default="results/full_3x/ces_within_party_blocks.csv")
    return p.parse_args()


def build_covariates(raw: pd.DataFrame) -> pd.DataFrame:
    """Recode the raw CES columns into the analysis covariates."""
    cov = pd.DataFrame(index=raw.index)
    age = SURVEY_YEAR - raw["birthyr"].astype(float)
    cov["age"] = age
    cov["age_sq"] = age ** 2
    cov["educ"] = raw["educ"]
    cov["region"] = raw["region"]
    cov["pew_religimp"] = raw["pew_religimp"]
    cov["pew_churatd"] = raw["pew_churatd"]
    cov["pew_bornagain"] = raw["pew_bornagain"]
    cov["gender4"] = raw["gender4"]
    # race: keep white/Black/Hispanic/Asian, everything else pooled
    cov["race5"] = raw["race"].where(raw["race"] <= 4, 5)
    # faminc_new 97 = "prefer not to say": keep as its own level rather than dropping
    cov["faminc_cat"] = raw["faminc_new"]
    cov["urbancity"] = raw["urbancity"]
    cov["marstat"] = raw["marstat"]
    cov["newsint"] = raw["newsint"]
    cov["pid7"] = raw["pid7"]
    cov["ideo5"] = raw["ideo5"]
    # categorical vars: missing becomes its own level so no respondent is dropped
    for c in cov.columns:
        if c not in ("age", "age_sq"):
            cov[c] = cov[c].fillna(-1).astype(int).astype("category")
    cov["age"] = cov["age"].fillna(cov["age"].median())
    cov["age_sq"] = cov["age_sq"].fillna(cov["age_sq"].median())
    return cov


def design_matrix(cov: pd.DataFrame, blocks: list[str]) -> np.ndarray:
    """One-hot the categorical blocks, standardise the continuous ones."""
    if not blocks:
        return np.zeros((len(cov), 0))
    cats, conts = [], []
    for b in blocks:
        bc, bn = BLOCKS[b]
        cats.extend(bc)
        conts.extend(bn)
    parts = []
    if cats:
        d = pd.get_dummies(cov[cats], drop_first=True)
        # drop levels that are constant in this subsample
        d = d.loc[:, d.nunique() > 1]
        parts.append(d.to_numpy(dtype=float))
    if conts:
        x = cov[conts].to_numpy(dtype=float)
        sd = x.std(axis=0)
        sd[sd == 0] = 1.0
        parts.append((x - x.mean(axis=0)) / sd)
    return np.hstack(parts) if parts else np.zeros((len(cov), 0))


def cv_predict(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """5-fold cross-validated weighted P(y=1). Intercept-only when X has no columns."""
    p = np.empty(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X if X.shape[1] else np.zeros((len(y), 1)), y):
        if X.shape[1] == 0:
            # weighted base rate of the training fold
            p[te] = float((y[tr] * w[tr]).sum() / w[tr].sum())
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def wmean(v: np.ndarray, w: np.ndarray) -> float:
    return float((v * w).sum() / w.sum())


def score(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    share = wmean(y, w)
    baseline = min(share, 1.0 - share)
    mismatch = wmean((y != (p > 0.5)).astype(float), w)
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        auc = np.nan
    # is the covariate-conditioned responder ever confident about an individual?
    confident = wmean(((p > 0.8) | (p < 0.2)).astype(float), w)
    return {
        "share_liberal": share,
        "baseline_mismatch": baseline,
        "mismatch": mismatch,
        "mismatch_reduction": baseline - mismatch,
        "auc": auc,
        "share_confident": confident,
        "p_iqr": float(np.subtract(*np.percentile(p, [75, 25]))),
    }


def main() -> int:
    args = parse_args()
    issues = pd.read_csv(args.issues)
    issues = issues[issues["analysis_tier"] == "main"].copy()
    issue_vars = issues["ces_variable"].tolist()
    print(f"{len(issue_vars)} main issues")

    raw = pd.read_stata(args.ces, columns=RAW_VARS + issue_vars, convert_categoricals=False)
    w_all = raw["commonweight"].astype(float)
    cov_all = build_covariates(raw)

    # per-respondent liberal side in {0, 1}; CES is forced-choice so there is no middle
    lib = pd.DataFrame(index=raw.index)
    for _, row in issues.iterrows():
        v = row["ces_variable"]
        sign = int(row["liberal_sign"])
        sup = set(leading_ints(row["ces_support_code"]))
        opp = set(leading_ints(row["ces_oppose_code"]))
        ans = raw[v]
        val = pd.Series(np.nan, index=raw.index)
        val[ans.isin(sup)] = 1.0 if sign > 0 else 0.0
        val[ans.isin(opp)] = 0.0 if sign > 0 else 1.0
        lib[v] = val

    rows, block_rows = [], []
    for party, code in PARTIES.items():
        in_party = (raw["pid3"] == code).to_numpy()
        for v in issue_vars:
            ok = in_party & lib[v].notna().to_numpy() & (w_all > 0).to_numpy()
            if ok.sum() < 200:
                continue
            y = lib[v].to_numpy()[ok].astype(int)
            w = w_all.to_numpy()[ok]
            cov = cov_all.loc[ok].reset_index(drop=True)
            if y.min() == y.max():
                continue

            for name, blocks in MODELS.items():
                X = design_matrix(cov, blocks)
                p = cv_predict(X, y, w)
                rows.append({"party": party, "issue": v, "model": name,
                             "n": int(ok.sum()), "n_features": X.shape[1],
                             **score(y, p, w)})

            for b in BLOCKS:
                X = design_matrix(cov, [b])
                p = cv_predict(X, y, w)
                s = score(y, p, w)
                block_rows.append({"party": party, "issue": v, "block": b,
                                   "auc": s["auc"], "mismatch": s["mismatch"],
                                   "baseline_mismatch": s["baseline_mismatch"]})
        print(f"  {party}: done")

    out = pd.DataFrame(rows)
    blocks_out = pd.DataFrame(block_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    blocks_out.to_csv(args.out_blocks, index=False)

    pd.set_option("display.width", 200)
    # equal weight per issue, matching the issue-clustered estimator elsewhere
    summ = (out.groupby(["party", "model"], observed=True)
            [["share_liberal", "baseline_mismatch", "mismatch",
              "mismatch_reduction", "auc", "share_confident", "p_iqr"]]
            .mean().reset_index())
    print("\n=== within-party predictability, averaged over issues ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    bsumm = (blocks_out.groupby(["party", "block"], observed=True)[["auc", "mismatch"]]
             .mean().reset_index().sort_values(["party", "auc"], ascending=[True, False]))
    print("\n=== single covariate blocks, CV AUC averaged over issues ===")
    print(bsumm.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nWrote {args.out}\nWrote {args.out_blocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
