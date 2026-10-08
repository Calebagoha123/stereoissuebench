#!/usr/bin/env python3
"""Does the stance scorer neutralise conservative writing more readily than liberal?

``judge_comparison.py`` rules out self-preference (a judge agreeing with humans
better on its own family's generations). It does not rule out a *directional*
failure: a scorer that is itself left-leaning might code a mildly conservative
paragraph as neutral more often than an equally mild liberal one. That failure
mode would manufacture exactly the neutral-hedging asymmetry the RQ1 composition
results rest on -- cues moving responses between the liberal and neutral piles
while conservative writing stays rare.

The 245-item gold supports a direct test. Every item carries ``liberal_sign``, so
both the human gold and each scorer can be put on a common liberal axis
(0 = fully conservative, 100 = fully liberal) rather than the raw
support-the-proposition axis. Then, per scorer:

  net lean            mean(scorer - gold) over all items; > 0 is a liberal tilt
  compression, cons.  mean(scorer - gold) among items humans placed conservative
                      (> 0 = pulled up toward the centre)
  compression, lib.   mean(gold - scorer) among items humans placed liberal
                      (> 0 = pulled down toward the centre)
  ASYMMETRY           compression(cons) - compression(lib); > 0 is the failure
                      mode the critique describes
  neutralisation      the same asymmetry after the 40/60 collapse: the share of
                      human-committed items the scorer calls neutral, by side.
                      This is the discrete quantity the composition figures use.

Two controls make the number interpretable rather than merely non-zero:

* every scorer is run, not only the one of record. DeBERTa is a fine-tuned encoder
  with no politics to have; if it shows the same asymmetry, the source is the
  construct and the annotators' codebook, not the judge's ideology.
* the two human annotators are scored against each other on the identical
  statistic, which gives the human--human noise floor. An asymmetry inside that
  band is not distinguishable from ordinary annotator disagreement.

CIs are percentile bootstrap over items on one shared resample matrix, so scorers
are paired and their intervals directly comparable.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

ROOT = __file__.rsplit("/analysis/", 1)[0]
OUT = f"{ROOT}/analysis/07_validation/out"
LO, HI = 40, 60          # same neutral band as judge_comparison.py
B, SEED = 5000, 7


def to_liberal(score: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Support-the-proposition (0-100) -> liberal axis (0-100)."""
    return np.where(sign > 0, score, 100.0 - score)


def stats_for(pred: np.ndarray, gold: np.ndarray) -> dict:
    """Directional-bias statistics on the liberal axis. NaN preds dropped."""
    ok = ~np.isnan(pred)
    p, g = pred[ok], gold[ok]
    con, lib = g < LO, g > HI
    e = p - g
    comp_con = e[con].mean() if con.any() else np.nan       # pulled toward centre
    comp_lib = -e[lib].mean() if lib.any() else np.nan      # pulled toward centre
    # discrete: share of human-committed items the scorer calls neutral
    neut = (p >= LO) & (p <= HI)
    nz_con = neut[con].mean() if con.any() else np.nan
    nz_lib = neut[lib].mean() if lib.any() else np.nan
    return {
        "n": len(p), "n_con": int(con.sum()), "n_lib": int(lib.sum()),
        "net_lean": e.mean(),
        "compress_con": comp_con, "compress_lib": comp_lib,
        "asymmetry": comp_con - comp_lib,
        "neutralise_con": nz_con, "neutralise_lib": nz_lib,
        "neutralise_gap": nz_con - nz_lib,
    }


KEYS = ["net_lean", "compress_con", "compress_lib", "asymmetry",
        "neutralise_con", "neutralise_lib", "neutralise_gap"]


def boot_cis(pred: np.ndarray, gold: np.ndarray, idx_mat: np.ndarray) -> dict:
    acc = {k: [] for k in KEYS}
    for idx in idx_mat:
        s = stats_for(pred[idx], gold[idx])
        for k in KEYS:
            acc[k].append(s[k])
    out = {}
    for k in KEYS:
        v = np.asarray(acc[k], float)
        v = v[np.isfinite(v)]
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
    return out


def main() -> int:
    # ---- gold, assembled exactly as in judge_comparison.py -------------------
    c = pd.read_csv(f"{ROOT}/annotation/ratings_caleb.csv")[["item_id", "score", "unratable"]]
    h = pd.read_csv(f"{ROOT}/annotation/ratings_hl.csv")[["item_id", "score", "unratable"]]
    keys = pd.read_csv(f"{OUT}/sample_keys.csv")[["item_id", "model", "liberal_sign",
                                                  "bert_pred_stance"]]
    m = (c.rename(columns={"score": "caleb", "unratable": "uc"})
         .merge(h.rename(columns={"score": "hl", "unratable": "uh"}), on="item_id")
         .merge(keys, on="item_id"))
    m = m[(m.uc == 0) & (m.uh == 0)].reset_index(drop=True)
    m["gold"] = (m.caleb + m.hl) / 2

    scorers = {"DeBERTa": "bert_pred_stance"}
    for path in sorted(glob.glob(f"{OUT}/judge_*.csv")):
        tag = path.rsplit("/judge_", 1)[1][:-4]
        if tag.startswith("smoke") or tag in ("comparison_metrics", "per_model_r"):
            continue
        j = pd.read_csv(path)
        col = f"judge_{tag}"
        m = m.merge(j[["item_id", "judge_score"]].rename(columns={"judge_score": col}),
                    on="item_id", how="left")
        scorers[tag] = col

    sign = m["liberal_sign"].to_numpy(float)
    gold_lib = to_liberal(m["gold"].to_numpy(float), sign)

    rng = np.random.default_rng(SEED)
    idx_mat = rng.integers(0, len(m), size=(B, len(m)))

    rows = []
    for name, col in scorers.items():
        pred_lib = to_liberal(m[col].to_numpy(float), sign)
        rows.append({"scorer": name, **stats_for(pred_lib, gold_lib),
                     **boot_cis(pred_lib, gold_lib, idx_mat)})

    # human--human noise floor: each annotator scored against the other
    for a, b in (("caleb", "hl"), ("hl", "caleb")):
        pa = to_liberal(m[a].to_numpy(float), sign)
        pb = to_liberal(m[b].to_numpy(float), sign)
        rows.append({"scorer": f"human:{a} vs {b}", **stats_for(pa, pb),
                     **boot_cis(pa, pb, idx_mat)})

    out = pd.DataFrame(rows)
    path = f"{OUT}/judge_ideological_lean.csv"
    out.to_csv(path, index=False)

    pd.set_option("display.width", 220)
    fmt = lambda x: f"{x:.2f}"
    print(f"n = {len(m)} gold items "
          f"({out.iloc[0]['n_con']} human-conservative, {out.iloc[0]['n_lib']} human-liberal)\n")
    print("=== continuous: points on the 0-100 liberal axis ===")
    show = out[["scorer", "net_lean", "net_lean_lo", "net_lean_hi",
                "compress_con", "compress_lib", "asymmetry",
                "asymmetry_lo", "asymmetry_hi"]]
    print(show.to_string(index=False, float_format=fmt))
    print("\n=== discrete: share of human-committed items called neutral (40-60 band) ===")
    show2 = out[["scorer", "neutralise_con", "neutralise_lib", "neutralise_gap",
                 "neutralise_gap_lo", "neutralise_gap_hi"]]
    print(show2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
