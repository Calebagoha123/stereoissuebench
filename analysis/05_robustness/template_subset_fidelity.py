#!/usr/bin/env python3
"""Positive control for the Arm-B template subset.

Arm B (rotated names/states) was generated on a genre-proportional subset of 35 of
the 145 writing templates, and Arm B is where the study's nulls live. That invites
an obvious objection: if those 35 templates happened to be *inert* -- unresponsive
to any identity cue -- a null under names would be an artifact of the instrument,
and the TOST equivalence intervals would inherit the artifact rather than detect
it (they are computed on the same 35 templates).

The control runs entirely inside Arm A, where the explicit label cues were
generated on all 145 templates. For each (model, explicit cue) we recompute the
shift twice from the *same responses*:

    full    cue and baseline over all 145 templates      (= model_shift_table)
    subset  cue and baseline over the 35 Arm-B templates

No regeneration and no cross-arm comparison is involved, so nothing here is
confounded with the fixed-vs-rotated instance design; the only thing that changes
is which templates enter the average.

If the subset reproduces the full-pool effects, the 35 templates demonstrably
register shifts far larger than the name/state effects being called null, which is
what licenses reading those nulls as substantive.

Uncertainty on the *difference* is a paired issue-clustered bootstrap: both
estimates are recomputed on the same resampled issues within a draw, so the
interval accounts for the fact that the subset estimate is nested inside the full
one and shares its responses. That dependence is why the difference must not be
read as an independent-samples test of "subset vs full" -- the quantity of interest
is the magnitude of the discrepancy, not a p-value against zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _common import (CUE_DISPLAY, CUE_ORDER, MODELS, ROBUST, load_all)  # noqa: E402

N_BOOT = 2000
SEED = 20260711

# Arm-A cues only: these are the ones generated over the full 145-template pool.
ARM_A_CUES = [(f, g) for f, g in CUE_ORDER if f.startswith("explicit")]


def _issue_stats(sub: pd.DataFrame, issues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-issue (sum, count) of y aligned to `issues`, zero-filled where absent."""
    g = sub.groupby("issue_id")["y"]
    sums = g.sum().reindex(issues).fillna(0.0).to_numpy()
    counts = g.count().reindex(issues).fillna(0).to_numpy(dtype=float)
    return sums, counts


def _boot_means(sums: np.ndarray, counts: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    """Clustered-bootstrap grand means: pool resampled issues with multiplicity.

    Identical arithmetic to _common._mean_over_issues, vectorised over draws.
    """
    s = sums[boot_idx].sum(axis=1)
    c = counts[boot_idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def main() -> None:
    df = load_all()
    rng = np.random.default_rng(SEED)
    recs = []

    for model in MODELS:
        d = df[df["model"] == model]
        base_all = d[d["cue_family"] == "baseline"]
        issues = np.sort(d["issue_id"].unique())
        boot_idx = rng.integers(0, len(issues), size=(N_BOOT, len(issues)))

        # The Arm-B template subset, read off the data rather than re-derived.
        sub_tmpl = np.sort(d.loc[d["arm"] == "B", "template_id"].unique())
        full_tmpl = np.sort(base_all["template_id"].unique())
        assert set(sub_tmpl) <= set(full_tmpl), "Arm-B templates not a subset of the pool"

        base_sub = base_all[base_all["template_id"].isin(sub_tmpl)]
        b_full = _issue_stats(base_all, issues)
        b_sub = _issue_stats(base_sub, issues)
        bm_full = _boot_means(*b_full, boot_idx)
        bm_sub = _boot_means(*b_sub, boot_idx)

        for fam, grp in ARM_A_CUES:
            cue_full = d[(d["cue_family"] == fam) & (d["cue_group"] == grp)]
            if cue_full.empty:
                continue
            cue_sub = cue_full[cue_full["template_id"].isin(sub_tmpl)]

            pt_full = cue_full["y"].mean() - base_all["y"].mean()
            pt_sub = cue_sub["y"].mean() - base_sub["y"].mean()

            cm_full = _boot_means(*_issue_stats(cue_full, issues), boot_idx)
            cm_sub = _boot_means(*_issue_stats(cue_sub, issues), boot_idx)
            dl_full = cm_full - bm_full
            dl_sub = cm_sub - bm_sub
            diff = dl_sub - dl_full          # paired within bootstrap draw

            ok = ~(np.isnan(dl_full) | np.isnan(dl_sub))
            dl_full, dl_sub, diff = dl_full[ok], dl_sub[ok], diff[ok]

            lo_f, hi_f = np.percentile(dl_full, [2.5, 97.5])
            lo_s, hi_s = np.percentile(dl_sub, [2.5, 97.5])
            lo_d, hi_d = np.percentile(diff, [2.5, 97.5])

            recs.append({
                "model": model, "cue_family": fam, "cue_group": grp,
                "cue_display": CUE_DISPLAY[(fam, grp)],
                "n_templates_full": len(full_tmpl), "n_templates_sub": len(sub_tmpl),
                "n_cue_full": len(cue_full), "n_cue_sub": len(cue_sub),
                "delta_full": pt_full, "se_full": float(np.std(dl_full, ddof=1)),
                "delta_full_lo": float(lo_f), "delta_full_hi": float(hi_f),
                "delta_sub": pt_sub, "se_sub": float(np.std(dl_sub, ddof=1)),
                "delta_sub_lo": float(lo_s), "delta_sub_hi": float(hi_s),
                "diff": pt_sub - pt_full, "diff_lo": float(lo_d), "diff_hi": float(hi_d),
                "abs_diff": abs(pt_sub - pt_full),
                # Does the subset interval cover the full-pool point estimate?
                "covers_full": bool(lo_s <= pt_full <= hi_s),
            })

    out = pd.DataFrame(recs)
    ROBUST.mkdir(parents=True, exist_ok=True)
    path = ROBUST / "template_subset_fidelity.csv"
    out.to_csv(path, index=False)

    pd.set_option("display.width", 220)
    print(out[["model", "cue_display", "delta_full", "delta_sub", "diff",
               "diff_lo", "diff_hi", "covers_full"]].to_string(index=False))

    r = np.corrcoef(out["delta_full"], out["delta_sub"])[0, 1]
    print(f"\n{len(out)} model x cue cells over {out.n_templates_sub.iloc[0]} of "
          f"{out.n_templates_full.iloc[0]} templates")
    print(f"correlation(full, subset)      r = {r:.4f}")
    print(f"max |subset - full|              = {out.abs_diff.max():.4f} "
          f"({out.loc[out.abs_diff.idxmax(), 'model']}, "
          f"{out.loc[out.abs_diff.idxmax(), 'cue_display']})")
    print(f"median |subset - full|           = {out.abs_diff.median():.4f}")
    print(f"subset CI covers full estimate   = {out.covers_full.sum()}/{len(out)}")
    print(f"sign agreement                   = "
          f"{int((np.sign(out.delta_full) == np.sign(out.delta_sub)).sum())}/{len(out)}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
