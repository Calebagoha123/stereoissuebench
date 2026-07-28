#!/usr/bin/env python3
"""Human IRR (Caleb vs HL) and DeBERTa performance on the 100-item validation set.

Gold = mean of the two annotators' 0-100 scores. IRR is measured between the two
annotators; DeBERTa (bert_pred_stance) is scored against the adjudicated gold.
3-class collapse uses the same 40/60 thresholds as the annotation guidelines.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix

ROOT = __file__.rsplit("/analysis/", 1)[0]
LO, HI = 40, 60  # against / neutral / for thresholds


def collapse(x):
    return np.where(x < LO, "against", np.where(x > HI, "for", "neutral"))


def icc_2_1(a, b):
    """ICC(2,1): two-way random, single rater, absolute agreement."""
    Y = np.column_stack([a, b]).astype(float)
    n, k = Y.shape
    gm = Y.mean()
    MSR = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)          # rows (subjects)
    MSC = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)          # cols (raters)
    SSE = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    MSE = SSE / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)


def krippendorff_interval(a, b):
    """Interval alpha, complete data, 2 coders per unit."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    N = len(a)
    Do = np.mean((a - b) ** 2)                                 # observed disagreement
    v = np.concatenate([a, b])
    m = len(v)
    De = (m * (v ** 2).sum() - v.sum() ** 2) / (m * (m - 1))   # expected disagreement
    return 1 - Do / De


def boot_ci(fn, *arrays, B=5000, seed=7, pct=(2.5, 97.5)):
    """Percentile bootstrap CI for fn(*arrays), resampling item pairs w/ replacement.

    Resamples paired rows (so both raters / pred+gold move together). Draws that
    make a statistic undefined (e.g. a constant column -> nan kappa) are dropped.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    stats = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        try:
            stats[i] = fn(*[a[idx] for a in arrays])
        except Exception:
            stats[i] = np.nan
    stats = stats[np.isfinite(stats)]
    return np.percentile(stats, pct[0]), np.percentile(stats, pct[1])


def ci_str(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


# ---- load ----------------------------------------------------------------
c = pd.read_csv(f"{ROOT}/annotation/ratings_caleb.csv")[["item_id", "score", "unratable"]]
h = pd.read_csv(f"{ROOT}/annotation/ratings_hl.csv")[["item_id", "score", "unratable"]]
keys = pd.read_csv(f"{ROOT}/analysis/07_validation/out/sample_keys.csv")

m = (c.rename(columns={"score": "caleb", "unratable": "u_c"})
      .merge(h.rename(columns={"score": "hl", "unratable": "u_h"}), on="item_id")
      .merge(keys[["item_id", "model", "bert_pred_stance"]], on="item_id"))
# drop any item either annotator flagged unratable
m = m[(m.u_c == 0) & (m.u_h == 0)].reset_index(drop=True)
m["gold"] = (m.caleb + m.hl) / 2
n = len(m)

# ---- 1. inter-annotator reliability -------------------------------------
diff = m.caleb - m.hl
pr, pp = pearsonr(m.caleb, m.hl)
sr, sp = spearmanr(m.caleb, m.hl)
icc = icc_2_1(m.caleb, m.hl)
alpha = krippendorff_interval(m.caleb, m.hl)
cc, hc = collapse(m.caleb), collapse(m.hl)
kap = cohen_kappa_score(cc, hc)
agree3 = np.mean(cc == hc)

# 95% bootstrap CIs (resample the n annotator pairs)
ca, ha = m.caleb.to_numpy(), m.hl.to_numpy()
pr_ci = boot_ci(lambda x, y: pearsonr(x, y)[0], ca, ha)
sr_ci = boot_ci(lambda x, y: spearmanr(x, y)[0], ca, ha)
icc_ci = boot_ci(icc_2_1, ca, ha)
alpha_ci = boot_ci(krippendorff_interval, ca, ha)
agree3_ci = boot_ci(lambda x, y: np.mean(collapse(x) == collapse(y)), ca, ha)
kap_ci = boot_ci(lambda x, y: cohen_kappa_score(collapse(x), collapse(y)), ca, ha)

print(f"=== Inter-annotator reliability (Caleb vs HL, n={n}) ===")
print(f"  Pearson r        {pr:.3f}  95%CI {ci_str(*pr_ci)}  (p={pp:.1e})")
print(f"  Spearman rho     {sr:.3f}  95%CI {ci_str(*sr_ci)}  (p={sp:.1e})")
print(f"  ICC(2,1)         {icc:.3f}  95%CI {ci_str(*icc_ci)}")
print(f"  Krippendorff a   {alpha:.3f}  95%CI {ci_str(*alpha_ci)}  (interval)")
print(f"  Mean signed diff {diff.mean():+.2f}   (Caleb - HL)")
print(f"  Mean abs diff    {diff.abs().mean():.2f}   (median {diff.abs().median():.1f})")
print(f"  3-class agree    {agree3:.1%}  95%CI {ci_str(*agree3_ci)}")
print(f"  Cohen's kappa    {kap:.3f}  95%CI {ci_str(*kap_ci)}  (3-class, 40/60)")

# ---- 2. DeBERTa vs gold --------------------------------------------------
d = m.dropna(subset=["bert_pred_stance"]).copy()
bpr, bpp = pearsonr(d.bert_pred_stance, d.gold)
bsr, bsp = spearmanr(d.bert_pred_stance, d.gold)
bmae = (d.bert_pred_stance - d.gold).abs().mean()
gold3, bert3 = collapse(d.gold), collapse(d.bert_pred_stance)
labels = ["against", "neutral", "for"]
acc = accuracy_score(gold3, bert3)
mf1 = f1_score(gold3, bert3, labels=labels, average="macro")
bkap = cohen_kappa_score(gold3, bert3, labels=labels)

# 95% bootstrap CIs (resample the pred/gold pairs)
dp, dg = d.bert_pred_stance.to_numpy(), d.gold.to_numpy()
bpr_ci = boot_ci(lambda x, y: pearsonr(x, y)[0], dp, dg)
bsr_ci = boot_ci(lambda x, y: spearmanr(x, y)[0], dp, dg)
bmae_ci = boot_ci(lambda x, y: np.abs(x - y).mean(), dp, dg)
acc_ci = boot_ci(lambda x, y: accuracy_score(collapse(y), collapse(x)), dp, dg)
mf1_ci = boot_ci(lambda x, y: f1_score(collapse(y), collapse(x), labels=labels, average="macro"), dp, dg)
bkap_ci = boot_ci(lambda x, y: cohen_kappa_score(collapse(y), collapse(x), labels=labels), dp, dg)

print(f"\n=== DeBERTa vs gold (mean of annotators, n={len(d)}) ===")
print(f"  Pearson r        {bpr:.3f}  95%CI {ci_str(*bpr_ci)}  (p={bpp:.1e})")
print(f"  Spearman rho     {bsr:.3f}  95%CI {ci_str(*bsr_ci)}  (p={bsp:.1e})")
print(f"  MAE (0-100)      {bmae:.2f}  95%CI {ci_str(*bmae_ci)}")
print(f"  3-class acc      {acc:.1%}  95%CI {ci_str(*acc_ci)}")
print(f"  Macro-F1         {mf1:.3f}  95%CI {ci_str(*mf1_ci)}")
print(f"  Cohen's kappa    {bkap:.3f}  95%CI {ci_str(*bkap_ci)}  (vs collapsed gold, 40/60)")
print("  Confusion (rows=gold, cols=DeBERTa) order against/neutral/for:")
cm = confusion_matrix(gold3, bert3, labels=labels)
print(pd.DataFrame(cm, index=[f"gold_{l}" for l in labels],
                   columns=[f"bert_{l}" for l in labels]).to_string())

# per-model DeBERTa correlation
print("\n  Per-model DeBERTa r vs gold:")
for mdl, g in d.groupby("model"):
    if len(g) > 2:
        r, _ = pearsonr(g.bert_pred_stance, g.gold)
        print(f"    {mdl:8s} n={len(g):3d}  r={r:.3f}  MAE={ (g.bert_pred_stance-g.gold).abs().mean():.1f}")

# ---- save ----------------------------------------------------------------
out = f"{ROOT}/analysis/07_validation/out/irr_deberta_metrics.csv"
NA = (np.nan, np.nan)  # metrics without a bootstrap CI
pd.DataFrame([
    ("irr", "n", n, *NA), ("irr", "pearson_r", pr, *pr_ci), ("irr", "spearman_rho", sr, *sr_ci),
    ("irr", "icc_2_1", icc, *icc_ci), ("irr", "krippendorff_alpha", alpha, *alpha_ci),
    ("irr", "mean_signed_diff", diff.mean(), *NA), ("irr", "mean_abs_diff", diff.abs().mean(), *NA),
    ("irr", "agree_3class", agree3, *agree3_ci), ("irr", "cohen_kappa_3class", kap, *kap_ci),
    ("deberta", "n", len(d), *NA), ("deberta", "pearson_r", bpr, *bpr_ci), ("deberta", "spearman_rho", bsr, *bsr_ci),
    ("deberta", "mae", bmae, *bmae_ci), ("deberta", "acc_3class", acc, *acc_ci),
    ("deberta", "macro_f1", mf1, *mf1_ci), ("deberta", "cohen_kappa_3class", bkap, *bkap_ci),
], columns=["block", "metric", "value", "ci_lo", "ci_hi"]).to_csv(out, index=False)
print(f"\nwrote {out}")
