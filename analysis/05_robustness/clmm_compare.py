#!/usr/bin/env python3
"""Compare the CLMM ordinal cue coefficients to the mean-difference Delta_k.

If the cumulative-link-mixed-model cue log-odds rank-order the same way as the
mean-difference cue effects, the RQ2 conclusions are invariant to treating the
{-1,0,+1} stance as interval rather than ordinal. Reports Spearman rank
correlation per model + the sign-agreement rate.

Reads results/robustness/clmm_coefs.csv (from clmm_robustness.R) and
results/robustness/model_shift_table.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, CUE_ORDER, CUE_DISPLAY, ROBUST


def main():
    clmm = pd.read_csv(ROBUST / "clmm_coefs.csv")
    mt = pd.read_csv(ROBUST / "model_shift_table.csv")
    # map the R cue label "family__group" back to (family, group)
    clmm[["cue_family", "cue_group"]] = clmm["cue"].str.split("__", n=1, expand=True)

    merged = clmm.merge(mt[["model", "cue_family", "cue_group", "cue_display", "model_shift"]],
                        on=["model", "cue_family", "cue_group"], how="inner")
    merged.to_csv(ROBUST / "clmm_vs_delta.csv", index=False)

    print("=== CLMM (ordinal) cue log-odds vs mean-difference Delta_k ===\n")
    for m in MODELS:
        sub = merged[merged.model == m]
        rho, p = stats.spearmanr(sub["clmm_logodds"], sub["model_shift"])
        r_pear, _ = stats.pearsonr(sub["clmm_logodds"], sub["model_shift"])
        sign_agree = (np.sign(sub["clmm_logodds"]) == np.sign(sub["model_shift"])).mean()
        print(f"{MODEL_LABEL[m]:>14}:  Spearman rho={rho:.3f} (p={p:.1e})  "
              f"Pearson r={r_pear:.3f}  sign-agreement={sign_agree*100:.0f}%")
    # pooled
    rho, p = stats.spearmanr(merged["clmm_logodds"], merged["model_shift"])
    print(f"{'pooled':>14}:  Spearman rho={rho:.3f} (p={p:.1e})")
    print(f"\nWrote {ROBUST/'clmm_vs_delta.csv'}")


if __name__ == "__main__":
    main()
