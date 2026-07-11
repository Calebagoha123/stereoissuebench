#!/usr/bin/env python3
"""RQ2 calibration-slope robustness: OLS-through-origin, free-intercept OLS, and
Deming (errors-in-variables) regression of model shift on real CES shift.

Motivation
----------
The headline RQ2 number is the slope of model stance shift (y) on real CES group
shift (x): slope 1 = calibrated personalisation, slope 0 = ignores real group
differences, 0<slope<1 = under-personalises. Three robustness moves:

1. **Free intercept.** The through-origin fit *assumes* a cue matching zero real
   difference produces zero model shift. We also fit the free-intercept model; a
   significantly nonzero intercept is itself a finding (a common shift applied to
   every cued condition, e.g. "any stored memory nudges liberal").

2. **Deming / errors-in-variables.** OLS assumes x is measured without error and
   attenuates the slope toward 0 when x is noisy. Both axes here are estimates
   (CES design/issue SE on x; bootstrap SE on y), so we report the Deming slope,
   which is unbiased given the ratio of error variances delta = var(y-err)/var(x-err).
   The Deming slope should sit above the OLS slope (de-attenuated).

3. **Non-independence.** The 14x3 points are not independent: each cue appears
   once per model at an identical x. We therefore (a) report **per-model** slopes
   (cleaner, and model heterogeneity is itself interesting) and (b) for the pooled
   fit, cluster the bootstrap on **cue** (resample the 14 cues, each carrying its
   3 model points).

Reads results/robustness/model_shift_table.csv (built by robustness_common.py).
Writes results/robustness/rq2_regression.csv and a figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _common import MODELS, MODEL_LABEL, MODEL_COLOUR, ROBUST
from _regression import ols_through_origin, ols_free, deming, x_var, Z


def fit_all(df):
    x = df["ces_shift_mean"].to_numpy()
    y = df["model_shift"].to_numpy()
    vy = df["model_shift_var"].to_numpy()
    vx = x_var(df).to_numpy()
    delta = np.mean(vy) / np.mean(vx)
    b_orig, se_orig = ols_through_origin(x, y)
    fr = ols_free(x, y)
    b_dem, a_dem = deming(x, y, delta)
    return {
        "n": len(x),
        "ols_origin_slope": b_orig, "ols_origin_slope_se": se_orig,
        "ols_free_slope": fr.slope, "ols_free_slope_se": fr.stderr,
        "ols_free_intercept": fr.intercept, "ols_free_intercept_se": fr.intercept_stderr,
        "ols_free_intercept_p": _intercept_p(fr, len(x)),
        "deming_slope": b_dem, "deming_intercept": a_dem, "deming_delta": delta,
        "mean_vx": float(np.mean(vx)), "mean_vy": float(np.mean(vy)),
    }


def _intercept_p(fr, n):
    t = fr.intercept / fr.intercept_stderr
    return float(2 * stats.t.sf(abs(t), df=n - 2))


def cluster_bootstrap_pooled(df, n_boot=5000, seed=7):
    """Resample the 14 cues (each carries its 3 model rows); refit each slope."""
    rng = np.random.default_rng(seed)
    cues = df.groupby(["cue_family", "cue_group"])
    groups = [g for _, g in cues]
    K = len(groups)
    out = {"ols_origin": [], "ols_free_slope": [], "ols_free_intercept": [], "deming": []}
    for _ in range(n_boot):
        idx = rng.integers(0, K, size=K)
        boot = pd.concat([groups[i] for i in idx], ignore_index=True)
        try:
            f = fit_all(boot)
        except Exception:
            continue
        out["ols_origin"].append(f["ols_origin_slope"])
        out["ols_free_slope"].append(f["ols_free_slope"])
        out["ols_free_intercept"].append(f["ols_free_intercept"])
        out["deming"].append(f["deming_slope"])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}
    return ci


def main():
    df = pd.read_csv(ROBUST / "model_shift_table.csv")
    rows = []

    # Pooled
    f = fit_all(df)
    ci = cluster_bootstrap_pooled(df)
    f.update({"scope": "pooled",
              "ols_origin_ci": ci["ols_origin"],
              "ols_free_slope_ci": ci["ols_free_slope"],
              "ols_free_intercept_ci": ci["ols_free_intercept"],
              "deming_ci": ci["deming"]})
    rows.append(f)

    # Per model
    for m in MODELS:
        sub = df[df["model"] == m]
        fm = fit_all(sub)
        fm["scope"] = m
        # simple point-resample bootstrap (1 point per cue within a model)
        rng = np.random.default_rng(hash(m) % 2**32)
        bo = {"ols_origin": [], "deming": [], "ols_free_slope": [], "ols_free_intercept": []}
        arr = sub.reset_index(drop=True)
        n = len(arr)
        for _ in range(5000):
            idx = rng.integers(0, n, size=n)
            b = arr.iloc[idx]
            try:
                ff = fit_all(b)
            except Exception:
                continue
            bo["ols_origin"].append(ff["ols_origin_slope"])
            bo["deming"].append(ff["deming_slope"])
            bo["ols_free_slope"].append(ff["ols_free_slope"])
            bo["ols_free_intercept"].append(ff["ols_free_intercept"])
        fm["ols_origin_ci"] = (np.percentile(bo["ols_origin"], 2.5), np.percentile(bo["ols_origin"], 97.5))
        fm["deming_ci"] = (np.percentile(bo["deming"], 2.5), np.percentile(bo["deming"], 97.5))
        fm["ols_free_slope_ci"] = (np.percentile(bo["ols_free_slope"], 2.5), np.percentile(bo["ols_free_slope"], 97.5))
        fm["ols_free_intercept_ci"] = (np.percentile(bo["ols_free_intercept"], 2.5), np.percentile(bo["ols_free_intercept"], 97.5))
        rows.append(fm)

    out = pd.DataFrame(rows)
    cols = ["scope", "n", "ols_origin_slope", "ols_origin_slope_se", "ols_origin_ci",
            "ols_free_slope", "ols_free_slope_se", "ols_free_slope_ci",
            "ols_free_intercept", "ols_free_intercept_se", "ols_free_intercept_p", "ols_free_intercept_ci",
            "deming_slope", "deming_intercept", "deming_delta", "deming_ci", "mean_vx", "mean_vy"]
    out = out[cols]
    out.to_csv(ROBUST / "rq2_regression.csv", index=False)
    pd.set_option("display.width", 240)

    def fmt(v):
        return f"[{v[0]:+.3f}, {v[1]:+.3f}]"

    print("=== RQ2 calibration slope: OLS(origin) vs OLS(free) vs Deming ===\n")
    for _, r in out.iterrows():
        print(f"[{r['scope']:>6}]  n={r['n']:.0f}")
        print(f"   OLS through-origin slope = {r['ols_origin_slope']:+.3f}  95%CI {fmt(r['ols_origin_ci'])}")
        print(f"   OLS free   slope         = {r['ols_free_slope']:+.3f}  95%CI {fmt(r['ols_free_slope_ci'])}")
        print(f"   OLS free   intercept     = {r['ols_free_intercept']:+.4f}  95%CI {fmt(r['ols_free_intercept_ci'])}  p={r['ols_free_intercept_p']:.3f}")
        print(f"   Deming     slope         = {r['deming_slope']:+.3f}  95%CI {fmt(r['deming_ci'])}  (delta={r['deming_delta']:.3f})")
        print()
    print(f"Wrote {ROBUST/'rq2_regression.csv'}")
    make_figure(df, out)


def make_figure(df, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    lim = 0.72
    xs = np.linspace(-lim, lim, 100)
    ax.plot(xs, xs, "--", color="#888", lw=1.1, zorder=1, label="calibration (y = x)")
    ax.axhline(0, color="#ccc", lw=0.7, zorder=0)
    ax.axvline(0, color="#ccc", lw=0.7, zorder=0)
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.errorbar(sub["ces_shift_mean"], sub["model_shift"],
                    yerr=[sub["model_shift"] - sub["model_shift_lo"], sub["model_shift_hi"] - sub["model_shift"]],
                    xerr=[sub["ces_shift_mean"] - sub["ces_shift_ci_low"], sub["ces_shift_ci_high"] - sub["ces_shift_mean"]],
                    fmt="o", ms=6, color=MODEL_COLOUR[m], ecolor=MODEL_COLOUR[m],
                    elinewidth=0.7, capsize=1.5, alpha=0.85, label=MODEL_LABEL[m], zorder=3)
    pooled = fits[fits["scope"] == "pooled"].iloc[0]
    ax.plot(xs, pooled["deming_slope"] * xs + pooled["deming_intercept"] if "deming_intercept" in pooled else pooled["deming_slope"] * xs,
            color="#111", lw=2.0, zorder=4,
            label=f"Deming slope = {pooled['deming_slope']:.2f}")
    ax.plot(xs, pooled["ols_origin_slope"] * xs, color="#111", lw=1.2, ls=":", zorder=4,
            label=f"OLS(origin) = {pooled['ols_origin_slope']:.2f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Real CES group shift  (subgroup − population)")
    ax.set_ylabel("Model stance shift  (cued − baseline)")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = Path("figures/robustness/rq2_regression.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
