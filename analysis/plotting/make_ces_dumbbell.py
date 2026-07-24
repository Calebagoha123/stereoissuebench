#!/usr/bin/env python3
"""Target-vs-model dumbbell: how far each model's cued shift lands from the REAL
CES-2025 subgroup shift, one small-multiple panel per model.

The comparison in one sentence: for every identity cue there is a real-world
number — how far that subgroup actually sits from the US population in CES 2025
(``ces_shift_mean``) — and a model number — how far the model's writing moved when
given the cue (``model_shift``). Both live on the same liberal-score axis, so the
question is simply *how far did the model land from reality?*

Idiom (why a dumbbell, not the calibration scatter). The scatter asks the reader to
read a regression-through-the-origin slope; the dumbbell shows the same fact as a
distance you can see. Per row:

    hollow marker ○  = CES subgroup reality (the target)
    filled  marker ● = where the model actually landed
    grey connector   = the miscalibration gap (the shift the model failed to make)

Because the filled marker hugs the zero line whenever the model under-reacts, the
whole figure reads at a glance: explicit party/identity cues track reality but fall
short (flattening); implicit name/state cues collapse onto zero regardless of how
far the real subgroup sits (the implicit-cue null). The CES target is identical
across models, so it is redrawn in every panel as each model's own reference.

Reads results/consolidated/01_master_cue_effects.csv (model × cue grain: has
model_shift[_lo/_hi] and ces_shift_mean[_ci_low/_ci_high] already).

Classifier of record: DeBERTa ``bert_liberal_score`` in {-1, 0, +1}. Titles are
omitted deliberately (a LaTeX caption carries the takeaway); each panel is headed
only by its model name.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Same model order / colour / shape vocabulary as the other thesis figures.
MODELS = ["qwen", "gemma", "llama", "gpt56terra", "sonnet5"]
MODEL_LABEL = {"qwen": "Qwen-3.6-27B", "gemma": "Gemma-3-12B", "llama": "Llama-3.1-8B",
               "gpt56terra": "GPT-5.6 Terra", "sonnet5": "Claude Sonnet 5"}
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#0072B2",
                "gpt56terra": "#CC79A7", "sonnet5": "#56B4E9"}
MODEL_MARKER = {"qwen": "o", "gemma": "s", "llama": "^", "gpt56terra": "D", "sonnet5": "v"}

CES_COLOUR = "#333333"      # the "reality" anchor: neutral, same in every panel
GAP_COLOUR = "#B9B9B9"      # the connector = the miscalibration gap

# Four cue types, most-direct politics signal first, matching Fig 1's row order.
# (band label, [(cue_family, cue_group, row label), ...])
BANDS = [
    ("PARTY\nLABEL", [
        ("explicit_political", "democrat", "Democrat"),
        ("explicit_political", "independent", "Independent"),
        ("explicit_political", "republican", "Republican")]),
    ("RACE ×\nGENDER", [
        ("explicit_demographic", "black_woman", "Black woman"),
        ("explicit_demographic", "black_man", "Black man"),
        ("explicit_demographic", "white_woman", "White woman"),
        ("explicit_demographic", "white_man", "White man")]),
    ("STATE", [
        ("implicit_political", "blue_state", "Blue state"),
        ("implicit_political", "swing_state", "Swing state"),
        ("implicit_political", "red_state", "Red state")]),
    ("NAME", [
        ("implicit_demographic", "black_woman", "Black woman"),
        ("implicit_demographic", "black_man", "Black man"),
        ("implicit_demographic", "white_woman", "White woman"),
        ("implicit_demographic", "white_man", "White man")]),
]


def _layout_rows(bands_spec):
    """Lay bands top->bottom with a gap between them. Returns rows
    (y, family, group, label), bands (label, y_center, y_lo, y_hi), y-extent."""
    rows, bands, y = [], [], 0.0
    for band, items in bands_spec:
        y0 = y
        for fam, grp, lbl in items:
            rows.append((y, fam, grp, lbl))
            y += 1.0
        bands.append((band, (y0 + y - 1.0) / 2.0, y0 - 0.5, y - 0.5))
        y += 0.9
    return rows, bands, y


def _key(df: pd.DataFrame) -> dict:
    """Index the consolidated table by (model, family, group) -> row."""
    out = {}
    for _, r in df.iterrows():
        out[(r.model, r.cue_family, r.cue_group)] = r
    return out


def _xlim(df: pd.DataFrame) -> tuple[float, float]:
    """Data-driven symmetric-ish x-range covering every marker + CI whisker, padded.
    No clamping: the point of the figure is the size of the gap, so nothing is hidden
    off-scale (the Republican shift is the widest and stays on-axis)."""
    cols = ["model_shift_lo", "model_shift_hi", "ces_shift_ci_low", "ces_shift_ci_high"]
    lo = float(np.nanmin(df[cols].to_numpy()))
    hi = float(np.nanmax(df[cols].to_numpy()))
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _direction_tags(ax):
    ax.text(0.0, 1.004, "←  more liberal", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8.5, color="#888888")
    ax.text(1.0, 1.004, "more conservative  →", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.5, color="#888888")


def load_slopes(path: Path) -> dict:
    """Per-model + pooled Deming (errors-in-variables) calibration slope (+95%
    cue-clustered bootstrap CI) from rq2_regression.py's output. Deming is the
    estimator of record here: both axes carry sampling error (CES design SE on x,
    bootstrap SE on y), so an errors-in-variables fit is the honest one. 1.0 =
    perfectly calibrated; < 1 = flattens the real gap."""
    import re
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        lo, hi = (float(v) for v in re.findall(r"-?\d+\.\d+", str(r.deming_ci))[:2])
        out[r.scope] = (float(r.deming_slope), lo, hi)
    return out


def _draw_panel(ax, key, model, rows, bands, xlim, ylabels: bool):
    colour = MODEL_COLOUR[model]
    marker = MODEL_MARKER[model]

    # Faint alternating band shading delimiting the four cue types (as in Fig 1).
    for i, (_, _, lo, hi) in enumerate(bands):
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#000000", alpha=0.04, zorder=0)
    ax.axvline(0, color="#222222", lw=1.4, zorder=1)  # the null (no shift)

    for yy, fam, grp, _ in rows:
        r = key[(model, fam, grp)]
        ces, ces_lo, ces_hi = r.ces_shift_mean, r.ces_shift_ci_low, r.ces_shift_ci_high
        mod, mod_lo, mod_hi = r.model_shift, r.model_shift_lo, r.model_shift_hi

        # Connector = the gap the reader is meant to see. Drawn first, under both
        # markers, so the dots read as the endpoints of a measured distance.
        ax.plot([ces, mod], [yy, yy], color=GAP_COLOUR, lw=2.4, solid_capstyle="round",
                zorder=2)

        # CES reality: hollow anchor + thin neutral whisker (survey sampling CI).
        ax.plot([ces_lo, ces_hi], [yy, yy], color=CES_COLOUR, lw=1.0, alpha=0.55, zorder=3)
        ax.plot(ces, yy, marker="o", ms=8.5, mfc="white", mec=CES_COLOUR, mew=1.6,
                ls="", zorder=5)

        # Model landing: filled model-coloured marker + bootstrap CI whisker.
        ax.plot([mod_lo, mod_hi], [yy, yy], color=colour, lw=1.7, alpha=0.85, zorder=4)
        ax.plot(mod, yy, marker=marker, ms=7.5, color=colour, mec="white", mew=0.8,
                ls="", zorder=6)

    ax.set_xlim(*xlim)
    ax.invert_xaxis()  # liberal on the LEFT, conservative on the RIGHT (matches Fig 1/3)
    ax.set_yticks([r[0] for r in rows])
    ax.set_ylim(max(r[0] for r in rows) + 0.7, min(r[0] for r in rows) - 0.7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    if ylabels:
        ax.set_yticklabels([r[3] for r in rows], fontsize=9.5)
    else:
        ax.tick_params(axis="y", labelleft=False)

    # Model name heads the panel (the only "title" — identity of the filled marker).
    ax.text(0.5, 1.045, MODEL_LABEL[model], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=11, fontweight="bold", color=colour)
    _direction_tags(ax)
    ax.set_xlabel(r"Shift in liberal score  ($\hat{\Delta}$)", fontsize=10)


def _band_labels(fig, ax_left, bands):
    """Cue-type band labels in the far-left gutter, aligned to each band's centre."""
    for band, yc, _, _ in bands:
        ax_left.text(-0.56, yc, band, transform=ax_left.get_yaxis_transform(),
                     ha="center", va="center", fontsize=8.5, fontweight="bold",
                     color="#666666", linespacing=0.9, rotation=0)


def fig_dumbbell(df: pd.DataFrame, out: Path, fmts) -> None:
    key = _key(df)
    rows, bands, _ = _layout_rows(BANDS)
    xlim = _xlim(df)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.05 * len(MODELS) + 1.1, 8.2),
                             squeeze=False, gridspec_kw={"wspace": 0.10})
    axes = axes[0]
    for j, m in enumerate(MODELS):
        _draw_panel(axes[j], key, m, rows, bands, xlim, ylabels=(j == 0))
    _band_labels(fig, axes[0], bands)

    # Figure-level legend: what the two markers and the connector mean. Model
    # identity is carried by the coloured panel headers, so the filled swatch is
    # shown neutral here (its shape/colour vary by panel).
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8.5, mfc="white", mec=CES_COLOUR,
                   mew=1.6, label="CES 2025 subgroup — real-world shift (target)"),
        plt.Line2D([], [], marker="o", ls="", ms=7.5, color="#666666", mec="white",
                   mew=0.8, label="Model shift (cued − no-cue baseline)"),
        plt.Line2D([], [], color=GAP_COLOUR, lw=2.4, solid_capstyle="round",
                   label="miscalibration gap"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.012))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.93, bottom=0.11)

    for ext in fmts:
        fig.savefig(out / f"fig_ces_dumbbell.{ext}", bbox_inches="tight")
    plt.close(fig)


PARTY = {"democrat", "independent", "republican"}


def _noparty_deming(shift_table: Path) -> dict:
    """Per-model + pooled Deming slope with the three party-label cues removed
    (+95% bootstrap CI, same clustering scheme as rq2_regression.py: cluster over
    cue-groups for pooled, point-resample cues for per-model). This is the RQ2
    robustness point the dumbbell can't show — how much of each model's calibration
    is carried by party labels alone. Fixed seeds, so the column is reproducible."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    from _regression import deming, x_var
    df = pd.read_csv(shift_table)
    keep = df[~((df.cue_family == "explicit_political") & (df.cue_group.isin(PARTY)))]

    def fit(d):
        x, y = d.ces_shift_mean.to_numpy(), d.model_shift.to_numpy()
        return deming(x, y, d.model_shift_var.mean() / x_var(d).mean())[0]

    def slope_ci(sub, pooled, seed, n_boot=5000):
        b = fit(sub)
        rng = np.random.default_rng(seed)
        samp = []
        if pooled:  # resample the cue-groups (each carries its per-model rows)
            groups = [g for _, g in sub.groupby(["cue_family", "cue_group"])]
            K = len(groups)
            for _ in range(n_boot):
                boot = pd.concat([groups[i] for i in rng.integers(0, K, K)], ignore_index=True)
                try: samp.append(fit(boot))
                except Exception: pass
        else:       # resample the cue points within the model
            arr = sub.reset_index(drop=True); n = len(arr)
            for _ in range(n_boot):
                try: samp.append(fit(arr.iloc[rng.integers(0, n, n)]))
                except Exception: pass
        return b, float(np.percentile(samp, 2.5)), float(np.percentile(samp, 97.5))

    out = {"pooled": slope_ci(keep, True, seed=7)}
    for i, m in enumerate(MODELS):
        out[m] = slope_ci(keep[keep.model == m], False, seed=1000 + i)
    return out


def write_slope_table(slopes: dict, shift_table: Path, out: Path) -> None:
    """Companion calibration-slope table (Markdown + LaTeX booktabs): the one-number
    summary the dumbbell trades away, per model AND pooled, in two columns —
    all cues vs. party labels removed — so the "party labels carry the calibration"
    result is visible for every model, not just the pool. Deming slopes; the all-cues
    column matches the figure's panel β exactly (frozen rq2_regression.csv)."""
    order = ["qwen", "gemma", "llama", "gpt56terra", "sonnet5", "pooled"]
    label = {**MODEL_LABEL, "pooled": "All models (pooled)"}
    npd = _noparty_deming(shift_table)
    rows = [(label[m], slopes[m], npd[m]) for m in order if m in slopes and m in npd]

    def cell(t):
        return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"

    md = ["| Model | β, all cues | β, party labels removed |", "|---|:--:|:--:|"]
    for name, allc, nop in rows:
        md.append(f"| {name} | {cell(allc)} | {cell(nop)} |")
    note = ("\n\nDeming (errors-in-variables) calibration slope; 95% CIs are cue-clustered "
            "bootstrap. β = 1 is perfect calibration; β < 1 means the model reproduces only "
            "that fraction of the real CES subgroup gap (flattening). The all-cues column "
            "matches each panel's header β in the figure. The gap between the two columns is "
            "the share of calibration carried by the three party-label cues — pooled, the "
            f"slope falls {rows[-1][1][0]:.2f} → {rows[-1][2][0]:.2f} once they are removed.")
    (out / "calibration_slope_table.md").write_text("\n".join(md) + note + "\n")

    def texcell(t):
        return rf"{t[0]:.2f}\,$[{t[1]:.2f},\,{t[2]:.2f}]$"

    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           r"Model & $\beta$ (all cues) & $\beta$ (party labels removed) \\",
           r"\midrule"]
    for name, allc, nop in rows:
        pre = r"\midrule " if name.startswith("All models") else ""
        tex.append(rf"{pre}{name} & {texcell(allc)} & {texcell(nop)} \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "calibration_slope_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md) + note)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/consolidated/01_master_cue_effects.csv")
    ap.add_argument("--slopes", default="results/robustness/rq2_regression.csv")
    ap.add_argument("--shift-table", default="results/robustness/model_shift_table.csv")
    ap.add_argument("--figures-dir", default="figures/ces_dumbbell")
    ap.add_argument("--format", default="both", choices=["pdf", "png", "both"])
    args = ap.parse_args()
    fmts = ["pdf", "png"] if args.format == "both" else [args.format]
    fd = Path(args.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table)
    df = df[df.model.isin(MODELS)].copy()
    fig_dumbbell(df, fd, fmts)
    write_slope_table(load_slopes(Path(args.slopes)), Path(args.shift_table), fd)
    print(f"\nWrote fig_ces_dumbbell + calibration_slope_table.{{md,tex}} to {fd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
