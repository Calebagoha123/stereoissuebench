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
        # Slanted cue labels: they take less horizontal room than upright text, so
        # the left gutter shrinks and the plotting area (the data) grows. The four
        # cue types are still read off the alternating band shading, so the
        # overarching band labels are dropped.
        ax.set_yticklabels([r[3] for r in rows], fontsize=10.5, rotation=28,
                           ha="right", va="center", rotation_mode="anchor")
    else:
        ax.tick_params(axis="y", labelleft=False)

    # Model name heads the panel (the only "title" — identity of the filled marker).
    ax.text(0.5, 1.045, MODEL_LABEL[model], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=11, fontweight="bold", color=colour)
    _direction_tags(ax)


# Symbols: the miscalibration gap δ_k names the x-axis; each of its two component
# shifts is spelled out on the legend entry for the marker that carries it.
X_LABEL = r"Shift from baseline  ($\hat{\Delta}_k$)"
CES_MATH = r"$\mu^{\mathrm{CES}}_{k} - \mu^{\mathrm{CES}}_{\mathrm{pop}}$"
MODEL_MATH = r"$\bar{Y}_k - \bar{Y}_{\mathrm{baseline}}$"


def fig_dumbbell(df: pd.DataFrame, out: Path, fmts) -> None:
    key = _key(df)
    rows, bands, _ = _layout_rows(BANDS)
    xlim = _xlim(df)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.05 * len(MODELS) + 0.7, 8.2),
                             squeeze=False, gridspec_kw={"wspace": 0.10})
    axes = axes[0]
    for j, m in enumerate(MODELS):
        _draw_panel(axes[j], key, m, rows, bands, xlim, ylabels=(j == 0))

    # Figure-level legend: each marker labelled with the shift it plots (CES real
    # shift vs. model shift); their difference is δ_k, the x-axis. Model identity is
    # carried by the coloured panel headers, so the filled swatch is neutral here.
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8.5, mfc="white", mec=CES_COLOUR,
                   mew=1.6, label=f"CES subgroup shift  ({CES_MATH})"),
        plt.Line2D([], [], marker="o", ls="", ms=7.5, color="#666666", mec="white",
                   mew=0.8, label=f"Model shift  ({MODEL_MATH})"),
        plt.Line2D([], [], color=GAP_COLOUR, lw=2.4, solid_capstyle="round",
                   label=r"miscalibration gap  ($\delta_k$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.02))
    left, right = 0.085, 0.99
    fig.subplots_adjust(left=left, right=right, top=0.93, bottom=0.135)
    fig.text((left + right) / 2.0, 0.082, X_LABEL, ha="center", va="center",
             fontsize=13, color="#1a1a1a")

    for ext in fmts:
        fig.savefig(out / f"fig_ces_dumbbell.{ext}", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Levels figure — per-issue absolute stance: model default vs. the US public
# --------------------------------------------------------------------------- #
# Companion to the dumbbell. The dumbbell compares SHIFTS (cue reactions); this
# compares LEVELS: on each issue, where does the model's no-cue default sit vs.
# where the American public actually sits (CES 2025)? It's the ceiling/floor
# context for flattening — a model already saturated on one side has no room to
# move for a same-side cue. Metric = liberal share AMONG OPINIONATED output, the
# only comparison that's fair to a survey with no "neutral" option; the model's
# (large) neutral-refusal rate is reported separately on the right.
import textwrap as _tw


def fig_levels(detail_csv: Path, out: Path, fmts) -> None:
    d = pd.read_csv(detail_csv)
    b = d[(d.cue_family == "baseline") & (d.model.isin(MODELS))]
    ces = b.groupby("issue").ces_lib_share.first()
    issues = ces.sort_values(ascending=False).index.tolist()  # most-liberal public on top
    yof = {iss: i for i, iss in enumerate(issues)}
    neutral = b.groupby("issue").neutral_rate.mean()  # mean across models (caveat strip)

    fig, ax = plt.subplots(figsize=(10.4, 9.2))
    for i in range(len(issues)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#000000", alpha=0.04, zorder=0)
    ax.axvline(0.5, color="#222222", lw=1.2, zorder=1)  # even lib/con split

    # Small vertical dodge so the five model markers stay distinct where they pile
    # up (many issues saturate all models near 100% liberal). The public anchor and
    # the cluster span stay on the row centre.
    dodge = dict(zip(MODELS, np.linspace(0.19, -0.19, len(MODELS))))
    for iss in issues:
        yy = yof[iss]
        c = float(ces[iss])
        sub = b[b.issue == iss]
        shares = [float(sub[sub.model == m].model_lib_share.iloc[0]) for m in MODELS
                  if (sub.model == m).any()]
        # faint span across the model cluster: its offset from the CES anchor is the
        # story (cluster left of anchor = models more liberal than the public).
        ax.plot([min(shares), max(shares)], [yy, yy], color="#E1E1E1", lw=2.2,
                solid_capstyle="round", zorder=2)
        for m in MODELS:
            r = sub[sub.model == m]
            if len(r):
                ax.plot(float(r.model_lib_share.iloc[0]), yy + dodge[m], marker=MODEL_MARKER[m],
                        ms=6.5, color=MODEL_COLOUR[m], mec="white", mew=0.7, ls="", zorder=4)
        ax.plot(c, yy, marker="o", ms=11, mfc="white", mec=CES_COLOUR, mew=1.8,
                ls="", zorder=5)  # the public anchor
        # mean neutral-refusal rate for this issue, at the right margin
        ax.text(1.045, yy, f"{neutral[iss]*100:.0f}%", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color="#999999")

    ax.set_xlim(-0.03, 1.03)  # pad so markers at 0%/100% aren't clipped at the spine
    ax.invert_xaxis()  # liberal share high -> plotted on the LEFT (matches other figs)
    ax.set_ylim(len(issues) - 0.6, -0.6)
    ax.set_yticks(range(len(issues)))
    ax.set_yticklabels([_tw.fill(iss, 34, break_long_words=False) for iss in issues],
                       fontsize=8.8, linespacing=0.9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%"])
    ax.set_xlabel("Liberal share of stance (among opinionated responses)", fontsize=11)
    ax.text(0.0, 1.006, "←  more liberal", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=9, color="#888888")
    ax.text(1.0, 1.006, "more conservative  →", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color="#888888")
    ax.text(1.045, 1.006, "neutral", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="#999999")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, mfc="white", mec=CES_COLOUR,
                          mew=1.8, label="US public (CES 2025)")]
    handles += [plt.Line2D([], [], marker=MODEL_MARKER[m], ls="", ms=7.5,
                           color=MODEL_COLOUR[m], mec="white", mew=0.7, label=MODEL_LABEL[m])
                for m in MODELS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.135),
              ncol=6, frameon=False, fontsize=9.5, columnspacing=1.3, handletextpad=0.4)
    fig.subplots_adjust(left=0.235, right=0.90, top=0.95, bottom=0.11)

    for ext in fmts:
        fig.savefig(out / f"fig_ces_levels.{ext}", bbox_inches="tight")
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
        return rf"{t[0]:.2f} $[{t[1]:.2f},\,{t[2]:.2f}]$"

    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Errors-in-variables calibration slopes by model}",
        r"\label{tab:calibration_slopes}",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        (r"\textbf{Model} & \shortstack{\textbf{All cues} \\ $\beta$ [95\% CI]} "
         r"& \shortstack{\textbf{Party labels removed} \\ $\beta$ [95\% CI]} \\"),
        r"\midrule",
    ]
    for name, allc, nop in rows:
        if name.startswith("All models"):
            tex.append(r"\midrule")
        tex.append(rf"{name} & {texcell(allc)} & {texcell(nop)} \\")
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{3pt}",
        r"\begin{minipage}{0.92\linewidth}",
        (r"\footnotesize\textit{Note:} Entries report Deming errors-in-variables "
         r"slopes with cue-clustered bootstrap 95\% confidence intervals. "
         r"$\beta=1$ denotes perfect calibration; $\beta<1$ indicates that model "
         r"shifts flatten the corresponding CES subgroup differences. The "
         r"right-hand column re-estimates each slope after excluding the Democrat, "
         r"Independent, and Republican cues."),
        r"\end{minipage}",
        r"\end{table}",
    ]
    (out / "calibration_slope_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md) + note)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/consolidated/01_master_cue_effects.csv")
    ap.add_argument("--detail", default="results/consolidated/08_issue_level_detail.csv")
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
    fig_levels(Path(args.detail), fd, fmts)
    write_slope_table(load_slopes(Path(args.slopes)), Path(args.shift_table), fd)
    print(f"\nWrote fig_ces_dumbbell + fig_ces_levels + calibration_slope_table.{{md,tex}} to {fd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
