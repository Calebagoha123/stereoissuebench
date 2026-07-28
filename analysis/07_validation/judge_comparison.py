#!/usr/bin/env python3
"""Compare every stance scorer against the adjudicated human gold.

Scores DeBERTa (bert_pred_stance) and each available LLM-as-judge
(analysis/07_validation/out/judge_*.csv) on the SAME 245-item gold (mean of the
two corrected annotators, unratable-flagged items dropped) with the SAME metrics
and 40/60 3-class collapse as irr_and_deberta.py. The point: DeBERTa should match
the human-agreement ceiling and beat the LLM judges.

Also reports a self-preference check -- per-generator-model judge agreement -- so
any inflation on a judge's own-family subset (Haiku vs sonnet5, a GPT judge vs
gpt56terra) is visible rather than hidden.
"""
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

ROOT = __file__.rsplit("/analysis/", 1)[0]
OUT = f"{ROOT}/analysis/07_validation/out"
LO, HI = 40, 60
LABELS = ["against", "neutral", "for"]


def collapse(x):
    return np.where(x < LO, "against", np.where(x > HI, "for", "neutral"))


METRIC_KEYS = ["pearson_r", "spearman_rho", "mae", "acc_3class", "macro_f1", "cohen_kappa"]


def metrics(pred, gold):
    p, g = np.asarray(pred, float), np.asarray(gold, float)
    ok = ~np.isnan(p)
    p, g = p[ok], g[ok]
    pr, _ = pearsonr(p, g)
    sr, _ = spearmanr(p, g)
    mae = np.abs(p - g).mean()
    pc, gc = collapse(p), collapse(g)
    return {
        "n": len(p), "pearson_r": pr, "spearman_rho": sr, "mae": mae,
        "acc_3class": accuracy_score(gc, pc),
        "macro_f1": f1_score(gc, pc, labels=LABELS, average="macro"),
        "cohen_kappa": cohen_kappa_score(gc, pc, labels=LABELS),
    }


def boot_metric_cis(pred, gold, idx_mat):
    """95% percentile CIs for each metric, over the shared resample matrix.

    idx_mat is (B, n): reusing one matrix across scorers makes the judge CIs
    paired (same resampled items), so overlaps are directly comparable.
    NaN preds are dropped per-draw; degenerate draws (undefined stat) are skipped.
    """
    p, g = np.asarray(pred, float), np.asarray(gold, float)
    acc = {k: [] for k in METRIC_KEYS}
    for idx in idx_mat:
        pi, gi = p[idx], g[idx]
        ok = ~np.isnan(pi)
        pi, gi = pi[ok], gi[ok]
        if len(pi) < 3:
            continue
        pc, gc = collapse(pi), collapse(gi)
        try:
            acc["pearson_r"].append(pearsonr(pi, gi)[0])
            acc["spearman_rho"].append(spearmanr(pi, gi)[0])
            acc["mae"].append(np.abs(pi - gi).mean())
            acc["acc_3class"].append(accuracy_score(gc, pc))
            acc["macro_f1"].append(f1_score(gc, pc, labels=LABELS, average="macro"))
            acc["cohen_kappa"].append(cohen_kappa_score(gc, pc, labels=LABELS))
        except Exception:
            continue
    out = {}
    for k in METRIC_KEYS:
        v = np.asarray(acc[k], float)
        v = v[np.isfinite(v)]
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(v, 2.5), np.percentile(v, 97.5)
    return out


# ---- gold + DeBERTa ----------------------------------------------------------
c = pd.read_csv(f"{ROOT}/annotation/ratings_caleb.csv")[["item_id", "score", "unratable"]]
h = pd.read_csv(f"{ROOT}/annotation/ratings_hl.csv")[["item_id", "score", "unratable"]]
keys = pd.read_csv(f"{OUT}/sample_keys.csv")[["item_id", "model", "bert_pred_stance"]]
m = (c.rename(columns={"score": "caleb", "unratable": "uc"})
     .merge(h.rename(columns={"score": "hl", "unratable": "uh"}), on="item_id")
     .merge(keys, on="item_id"))
m = m[(m.uc == 0) & (m.uh == 0)].reset_index(drop=True)
m["gold"] = (m.caleb + m.hl) / 2

scorers = {"DeBERTa": "bert_pred_stance"}

# ---- attach judge files ------------------------------------------------------
for path in sorted(glob.glob(f"{OUT}/judge_*.csv")):
    tag = path.rsplit("/judge_", 1)[1][:-4]
    if tag.startswith("smoke") or tag in ("comparison_metrics", "per_model_r"):
        continue
    j = pd.read_csv(path)
    col = f"judge_{tag}"
    m = m.merge(j[["item_id", "judge_score"]].rename(columns={"judge_score": col}),
                on="item_id", how="left")
    scorers[tag] = col

# ---- comparison table --------------------------------------------------------
# one shared resample matrix -> paired CIs across scorers (overlaps comparable)
B, seed = 5000, 7
rng = np.random.default_rng(seed)
idx_mat = rng.integers(0, len(m), size=(B, len(m)))

rows = []
for name, col in scorers.items():
    r = metrics(m[col], m.gold)
    r.update(boot_metric_cis(m[col], m.gold, idx_mat))
    r["scorer"] = name
    rows.append(r)
ci_cols = [f"{k}_{b}" for k in METRIC_KEYS for b in ("lo", "hi")]
tbl = pd.DataFrame(rows).set_index("scorer")[
    ["n", "pearson_r", "spearman_rho", "mae", "acc_3class", "macro_f1", "cohen_kappa"] + ci_cols]

# human ceiling for reference (with CI, same resamples)
hr, _ = pearsonr(m.caleb, m.hl)
ca, ha = m.caleb.to_numpy(), m.hl.to_numpy()
hr_boot = [pearsonr(ca[i], ha[i])[0] for i in idx_mat]
hr_lo, hr_hi = np.percentile(hr_boot, 2.5), np.percentile(hr_boot, 97.5)
print(f"Human ceiling (Caleb vs HL, n={len(m)}): Pearson r = {hr:.3f}  95%CI [{hr_lo:.3f}, {hr_hi:.3f}]\n")
print("=== Scorer vs adjudicated gold (Pearson r with 95% bootstrap CI) ===")
disp = tbl.assign(pearson_ci=[f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(tbl.pearson_r_lo, tbl.pearson_r_hi)])
print(disp[["n", "pearson_r", "pearson_ci", "cohen_kappa", "macro_f1", "mae"]].round(3).to_string())
tbl.to_csv(f"{OUT}/judge_comparison_metrics.csv")

# ---- per-generator-model r (self-preference visibility) ----------------------
print("\n=== Per-generator-model Pearson r vs gold ===")
per = {}
for name, col in scorers.items():
    row = {}
    for mdl, g in m.groupby("model"):
        gg = g.dropna(subset=[col])
        row[mdl] = pearsonr(gg[col], gg.gold)[0] if len(gg) > 2 else np.nan
    per[name] = row
per_tbl = pd.DataFrame(per).T
print(per_tbl.round(3).to_string())
print("\nSelf-preference watch: Haiku judge shares the Claude family with sonnet5;")
print("a GPT judge shares it with gpt56terra. A judge that agrees with gold *more*")
print("on its own-family column than elsewhere would show inflation there.")
per_tbl.to_csv(f"{OUT}/judge_per_model_r.csv")
print(f"\nwrote {OUT}/judge_comparison_metrics.csv and judge_per_model_r.csv")
