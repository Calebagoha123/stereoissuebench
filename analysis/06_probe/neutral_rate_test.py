#!/usr/bin/env python3
"""Are the models' opinion predictions less neutral than their writing? (§4.4)

The RQ3 text claims the predictions are polarised while the writing hedges to neutral
far more often -- i.e. under-writing shows up as a shift in *composition*, not only in
mean magnitude. That is a comparison of two proportions, and it is currently asserted
without a test. This script supplies one.

WHICH TEST AND WHY. Week 3 covers tests for proportions directly, and the outcome here
is exactly a proportion: the share of responses that are neutral. Two wrinkles decide
the implementation:

  * The two proportions are NOT independent samples. The same cue x issue cells produce
    both a prediction and a written response, so a two-sample z-test for proportions
    (which assumes independent groups) is the wrong null. We pair on the issue instead:
    within each issue, compute the neutral rate on each side, take the difference, then
    run a one-sample t-interval on the 19 issue-level differences. This is the same
    clustered estimator used everywhere else in the thesis (Week 8: clustered SEs treat
    each unit as an independent block of information; df come from the 19 clusters).

  * "Neutral" must mean the same thing on both sides or the comparison is vacuous. The
    written side is neutral when the classifier of record assigns the middle category
    (luna_liberal_disc == 0), which is the 0-100 judge score falling in the [40, 60]
    band. The prediction side is a 0-100 number on the same support, so we apply the
    SAME [40, 60] band to it. Any band would do provided it is common to both sides;
    the point of matching them is that the band cancels from the comparison.

We report the paired difference with a t-interval, and additionally the unpaired
two-proportion z-test (Week 3's textbook form) so the reader can see that the
dependence correction is what it is rather than having to take it on trust.

Usage:  python3 analysis/06_probe/neutral_rate_test.py [--band 40 60]
Writes: results/probe_internal/neutral_rate_test.{csv,md}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, load_model  # noqa: E402

BELIEF = Path("results/full")
OUT = Path("results/probe_internal")


def per_issue_neutral_rates(model: str, lo: float, hi: float) -> pd.DataFrame:
    """Neutral share on each side, per issue, for one model (cued conditions only).

    Baseline rows are excluded from both sides: the claim is about how the models write
    *when conditioned on a cue* versus what they predict for that same cue."""
    # --- written side: middle category of the classifier of record
    w = load_model(model)
    w = w[w.cue_family != "baseline"]
    wr = w.assign(neu=(w["y"] == 0).astype(float)).groupby("issue_id")["neu"].mean()

    # --- prediction side: same 0-100 band applied to the elicited estimate
    p = pd.read_csv(BELIEF / f"belief_probe_{model}.csv", low_memory=False)
    p = p[(p.probe_kind == "opinion") & (p.cue_family != "baseline")].copy()
    p["s"] = pd.to_numeric(p["parsed_score"], errors="coerce")
    p = p.dropna(subset=["s"])
    p["neu"] = ((p["s"] >= lo) & (p["s"] <= hi)).astype(float)
    pr = p.groupby("issue_id")["neu"].mean()

    idx = wr.index.intersection(pr.index)
    return pd.DataFrame({"written_neutral": wr.loc[idx], "predicted_neutral": pr.loc[idx]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", nargs=2, type=float, default=[40.0, 60.0],
                    help="neutral band on the 0-100 scale, applied to BOTH sides")
    args = ap.parse_args()
    lo, hi = args.band
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in MODELS:
        r = per_issue_neutral_rates(m, lo, hi)
        d = r.written_neutral - r.predicted_neutral        # paired, one value per issue
        n = len(d)
        mu = d.mean()
        se = d.std(ddof=1) / np.sqrt(n)
        t = mu / se
        ci = stats.t.interval(0.95, n - 1, mu, se)

        # Week 3's unpaired two-proportion z, on the raw response counts, for contrast.
        w = load_model(m)
        w = w[w.cue_family != "baseline"]
        p = pd.read_csv(BELIEF / f"belief_probe_{m}.csv", low_memory=False)
        p = p[(p.probe_kind == "opinion") & (p.cue_family != "baseline")].copy()
        p["s"] = pd.to_numeric(p["parsed_score"], errors="coerce")
        p = p.dropna(subset=["s"])
        n1, x1 = len(w), int((w["y"] == 0).sum())
        n2, x2 = len(p), int(((p["s"] >= lo) & (p["s"] <= hi)).sum())
        p1, p2 = x1 / n1, x2 / n2
        pool = (x1 + x2) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))

        rows.append(dict(model=m, n_issues=n,
                         written_neutral=r.written_neutral.mean(),
                         predicted_neutral=r.predicted_neutral.mean(),
                         diff=mu, lo=ci[0], hi=ci[1], t=t,
                         p_paired=2 * stats.t.sf(abs(t), n - 1),
                         z_unpaired=z, p_unpaired=2 * stats.norm.sf(abs(z)),
                         n_written=n1, n_predicted=n2))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "neutral_rate_test.csv", index=False)

    md = [f"## Neutral rate: written stance vs elicited prediction "
          f"(band [{lo:.0f}, {hi:.0f}] on both sides)", "",
          "Paired on issue (19 clusters, df = 18); cued conditions only. A positive "
          "difference means the writing is neutral more often than the prediction is.", "",
          "| Model | written neutral | predicted neutral | difference [95% CI] | t | p (paired) | p (unpaired z) |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for _, r in res.iterrows():
        md.append(f"| {MODEL_LABEL[r.model]} | {r.written_neutral:.3f} | {r.predicted_neutral:.3f} "
                  f"| {r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.t:.1f} "
                  f"| {r.p_paired:.2e} | {r.p_unpaired:.2e} |")
    md += ["",
           "The unpaired two-proportion $z$ is reported only for comparison: it treats the "
           "two sides as independent samples, which they are not (the same cue $\\times$ "
           "issue cells generate both), so it overstates precision. The paired "
           "issue-clustered interval is the one to quote.", "",
           "### Verdict on the claim that writing hedges neutral more than prediction does",
           ""]
    support, null, reverse = [], [], []
    for _, r in res.iterrows():
        lab = MODEL_LABEL[r.model]
        if r.lo > 0:
            support.append(f"{lab} ({r['diff']:+.2f})")
        elif r.hi < 0:
            reverse.append(f"{lab} ({r['diff']:+.2f})")
        else:
            null.append(f"{lab} ({r['diff']:+.2f})")
    md += [f"- **supports** the claim ({len(support)}): " + (", ".join(support) or "none"),
           f"- **no difference** ({len(null)}): " + (", ".join(null) or "none"),
           f"- **reverses** the claim ({len(reverse)}): " + (", ".join(reverse) or "none"), ""]
    if reverse or null:
        md += ["The claim as currently written in §4.4 (\"the predictions are polarised "
               "while the writing hedges to neutral far more often\") is therefore **not "
               "supportable as a blanket statement across models** and must be scoped to "
               "the models where it holds. Note in particular that a model can show "
               "$\\beta < 1$ (under-writing in magnitude) without hedging its writing more "
               "than its prediction: the magnitude shortfall and the neutral-composition "
               "shift are separate phenomena, and only the former is general here.", ""]
    (OUT / "neutral_rate_test.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {OUT}/neutral_rate_test.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
