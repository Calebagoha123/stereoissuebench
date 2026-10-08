#!/usr/bin/env python3
"""Cue interchangeability and group differentiation, in the Tonneau et al. idiom.

\\citet{tonneau_demographic_2026} test whether cues for the same group are
interchangeable by correlating *cue-induced deviation vectors* -- per item, the mean
outcome under a cue minus the mean outcome with no cue -- and reading two contrasts
off the resulting matrix:

  cross-cue, same group    does a label and a name move the same items the same way?
  cross-group, same cue    does cueing a different group move different items?

Their headline is that the second exceeds the first: how identity is cued matters more
than which group is cued. We run the same two contrasts on the political-writing
outcome, which is where the comparison stops being free.

Their deviation vectors are reliable because each item-cell averages ~150 responses
(50 names per subgroup x 3 seeds). Ours average ~10. A correlation between two noisy
vectors is attenuated by the square root of the product of their reliabilities, so the
raw numbers here are not comparable to theirs and must not be read as evidence that
cues diverge: a ceiling of 0.5 looks like divergence whatever the truth is. We
therefore report, for every pair, the observed r, the split-half reliability of each
vector (Spearman-Brown corrected to full length), the implied ceiling, and the
disattenuated r -- which is the only column comparable to Tonneau's.

Items are issue x template-bin cells, restricted to the template grid the sampled-cue
arm (B) shares with the fixed-cue arm (A), so every cue is measured on identical items.

    python analysis/05_robustness/cue_interchangeability.py
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CUES = {
    "party": ("explicit_political", ["democrat", "republican", "independent"]),
    "race_gender": ("explicit_demographic", ["black_woman", "black_man",
                                             "white_woman", "white_man"]),
    "state": ("implicit_political", ["blue_state", "red_state", "swing_state"]),
    "name": ("implicit_demographic", ["black_woman", "black_man",
                                      "white_woman", "white_man"]),
}
# The two families that cue the same four groups by different means: the pair the
# cross-cue contrast is actually about.
SAME_GROUP_PAIRS = [("race_gender", "name")]


def load(path: Path, n_bins: int) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    parts = d["prompt_id"].str.split("__", expand=True)
    d["tpl"] = parts[1]
    d["y"] = pd.to_numeric(d["bert_liberal_score"], errors="coerce")
    d = d.dropna(subset=["y"])
    # Restrict to the item grid arm B actually covers, so all four cue families are
    # measured on the same items rather than on overlapping but unequal template sets.
    shared = set(map(tuple, d.loc[d.arm == "B", ["issue_id", "tpl"]]
                     .drop_duplicates().to_numpy()))
    d = d[[t in shared for t in zip(d.issue_id, d.tpl)]].copy()
    tpls = sorted(d.tpl.unique())
    d["bin"] = d.tpl.map({t: i % n_bins for i, t in enumerate(tpls)})
    d["item"] = d.issue_id.astype(str) + "|" + d["bin"].astype(str)
    return d


def dev_vector(d: pd.DataFrame, fam: str, groups: list[str], half: int) -> pd.Series:
    """Mean outcome under the cue minus mean outcome with no cue, per item.

    The baseline is taken from the *same* half as the cue, so that two vectors built
    from opposite halves share no responses at all -- neither cue-side nor baseline-
    side. Subtracting one common baseline estimate from both vectors would make them
    correlate through that shared noise alone, which at our cell sizes dominates
    everything else (it puts every disattenuated correlation above 1).
    """
    sub = d[d.half == half]
    cue = sub[sub.cue_family.eq(fam) & sub.cue_group.isin(groups)]
    base = sub[sub.cue_family.eq("baseline")]
    return (cue.groupby("item").y.mean() - base.groupby("item").y.mean()).dropna()


def _corr(a: pd.Series, b: pd.Series) -> tuple[float, pd.DataFrame]:
    j = pd.concat([a.rename("x"), b.rename("y")], axis=1).dropna()
    return j.x.corr(j.y), j


def paired_corr(d, lf, lg, rf, rg):
    """Cross-half correlation between two cues, and each cue's own split-half.

    r_obs is the average of corr(x_A, y_B) and corr(x_B, y_A): the two vectors never
    share a response, so no common noise inflates it. Each cue's split-half
    reliability corr(x_A, x_B) is on the same half-length footing, so the ratio
    r_obs / sqrt(rel_x * rel_y) is the standard disattenuated correlation with no
    Spearman-Brown step needed.
    """
    xA, xB = dev_vector(d, lf, lg, 0), dev_vector(d, lf, lg, 1)
    yA, yB = dev_vector(d, rf, rg, 0), dev_vector(d, rf, rg, 1)
    r1, j1 = _corr(xA, yB)
    r2, _ = _corr(xB, yA)
    rel_x, _ = _corr(xA, xB)
    rel_y, _ = _corr(yA, yB)
    return float(np.mean([r1, r2])), float(rel_x), float(rel_y), j1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/full_3x")
    ap.add_argument("--out", default="results/robustness/cue_interchangeability.csv")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rows = []
    for path in sorted(Path(a.results).glob("bert_eval_*.csv")):
        model = path.stem.replace("bert_eval_", "")
        d = load(path, a.bins)
        d["half"] = d.groupby(["item", "cue_family", "cue_group"]).cumcount() % 2

        # Same-group-across-cues, then all group pairs within each cue family.
        specs = [("cross_cue_same_group", f"{l} vs {r}",
                  CUES[l][0], CUES[l][1], CUES[r][0], CUES[r][1])
                 for l, r in SAME_GROUP_PAIRS]
        for cue, (fam, gs) in CUES.items():
            specs += [("cross_group_same_cue", f"{cue}: {g1} vs {g2}",
                       fam, [g1], fam, [g2]) for g1, g2 in combinations(gs, 2)]

        for kind, lab, lf, lg, rf, rg in specs:
            r, rel_x, rel_y, j = paired_corr(d, lf, lg, rf, rg)
            ceil = float(np.sqrt(rel_x * rel_y)) if rel_x > 0 and rel_y > 0 else np.nan
            # Cluster bootstrap over issues, matching every other CI in the thesis.
            iss = np.array([i.split("|")[0] for i in j.index])
            uniq = np.unique(iss)
            xv, yv = j.x.to_numpy(), j.y.to_numpy()
            draws = []
            for _ in range(2000):
                take = rng.integers(0, len(uniq), len(uniq))
                idx = np.concatenate([np.where(iss == uniq[t])[0] for t in take])
                if xv[idx].std() > 0 and yv[idx].std() > 0:
                    draws.append(np.corrcoef(xv[idx], yv[idx])[0, 1])
            draws = np.asarray(draws)
            rows.append({"model": model, "contrast": kind, "pair": lab,
                         "n_items": len(j), "r": r,
                         "r_lo": np.percentile(draws, 2.5),
                         "r_hi": np.percentile(draws, 97.5),
                         "rel_left": rel_x, "rel_right": rel_y, "ceiling": ceil,
                         "r_disattenuated": r / ceil if ceil and ceil > 0 else np.nan})
        print(f"{model}: {len(d)} responses, {d.item.nunique()} items", flush=True)

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    print(f"\nwrote {a.out}\n")
    for kind in ("cross_cue_same_group", "cross_group_same_cue"):
        s = out[out.contrast.eq(kind)]
        print(f"== {kind} ==")
        print(s.groupby("pair")[["r", "ceiling", "r_disattenuated"]]
              .mean().round(3).to_string())
        print()


if __name__ == "__main__":
    main()
