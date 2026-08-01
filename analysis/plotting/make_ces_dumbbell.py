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
import matplotlib.patheffects as _pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style
from _markers import marker_ms as MS  # equal-ink marker sizing (see _markers.py)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
_style.apply(plt)  # Computer Modern, to match the thesis document

# Same model order / colour / shape vocabulary as the other thesis figures.
MODELS = ["qwen", "gemma", "llama", "gpt56terra", "sonnet5"]
MODEL_LABEL = {"qwen": "Qwen-3.6-27B", "gemma": "Gemma-3-12B", "llama": "Llama-3.1-8B",
               "gpt56terra": "GPT-5.6 Terra", "sonnet5": "Claude Sonnet 5"}
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#0072B2",
                "gpt56terra": "#CC79A7", "sonnet5": "#56B4E9"}
MODEL_MARKER = {"qwen": "o", "gemma": "s", "llama": "^", "gpt56terra": "D", "sonnet5": "v"}

CES_COLOUR = "#333333"      # the "reality" anchor: neutral, same in every panel
GAP_COLOUR = "#B9B9B9"      # the connector = the miscalibration gap

# White halo behind the CES target so it stays legible where a model marker lands
# on it (used in every CES figure, hence module level).
XHALO = [_pe.withStroke(linewidth=1.8, foreground="white")]

# Below this share of responses taking a side, the directional share is not drawn
# at all. Half is the natural cut: past it, the majority of the cell's responses
# are not represented by the marker, so the marker is more misleading than absent.
COMMIT_MIN = 0.50

# Four cue types, most-direct politics signal first, matching Fig 1's row order.
# (band label, [(cue_family, cue_group, row label), ...])
BANDS = [
    ("PARTY\nLABEL", [
        ("explicit_political", "democrat", "Democrat"),
        ("explicit_political", "independent", "Independent"),
        ("explicit_political", "republican", "Republican")]),
    ("RACE " + _style.TIMES + "\nGENDER", [
        ("explicit_demographic", "black_woman", "Black woman"),
        ("explicit_demographic", "black_man", "Black man"),
        ("explicit_demographic", "white_woman", "White woman"),
        ("explicit_demographic", "white_man", "White man")]),
    ("STATE", [
        ("implicit_political", "blue_state", "Blue state"),
        ("implicit_political", "swing_state", "Swing state"),
        ("implicit_political", "red_state", "Red state")]),
    ("NAME", [
        ("implicit_demographic", "black_woman", "Name: Black woman"),
        ("implicit_demographic", "black_man", "Name: Black man"),
        ("implicit_demographic", "white_woman", "Name: White woman"),
        ("implicit_demographic", "white_man", "Name: White man")]),
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
    ax.text(0.0, 1.004, _style.ARROW_L + "  more liberal", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8.5, color="#888888")
    ax.text(1.0, 1.004, "more conservative  " + _style.ARROW_R, transform=ax.transAxes, ha="right",
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
            ax.axhspan(lo, hi, color="#5A5A66", alpha=0.045, zorder=0)
    ax.axvline(0, color="#777777", lw=1.2, zorder=1)  # the null (no shift)

    for yy, fam, grp, _ in rows:
        r = key[(model, fam, grp)]
        ces, ces_lo, ces_hi = r.ces_shift_mean, r.ces_shift_ci_low, r.ces_shift_ci_high
        mod, mod_lo, mod_hi = r.model_shift, r.model_shift_lo, r.model_shift_hi

        # Connector = the gap the reader is meant to see. Drawn first, under both
        # markers, so the dots read as the endpoints of a measured distance.
        ax.plot([ces, mod], [yy, yy], color=GAP_COLOUR, lw=2.4, solid_capstyle="round",
                zorder=2)

        # Same target/landing scheme as fig_ces_party_levels: soft interval bands,
        # a dark haloed X for the CES target, a filled model-coloured marker for
        # where the model landed.
        ax.plot([ces_lo, ces_hi], [yy, yy], color=CES_COLOUR, lw=2.0, alpha=0.45,
                zorder=3, solid_capstyle="round")
        ax.plot([mod_lo, mod_hi], [yy, yy], color=colour, lw=2.0, alpha=0.40,
                zorder=4, solid_capstyle="round")
        # X under the model marker here (the reverse of the party-levels figure):
        # on the implicit-cue rows the model lands almost exactly on the target, and
        # the larger X would otherwise swallow the marker whose position is the point.
        ax.plot(ces, yy, marker="X", ms=MS("X", 5.8), mfc=CES_COLOUR, mec=CES_COLOUR,
                mew=0.5, ls="", zorder=5, path_effects=XHALO)
        ax.plot(mod, yy, marker=marker, ms=MS(marker, 6.0), color=colour, mec="white",
                mew=0.8, ls="", zorder=6)

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
        plt.Line2D([], [], marker="X", ls="", ms=MS("X", 5.8), color=CES_COLOUR,
                   label=f"CES subgroup shift  ({CES_MATH})"),
        plt.Line2D([], [], marker="o", ls="", ms=MS("o", 6.0), color="#666666",
                   mec="white", mew=0.8, label=f"Model shift  ({MODEL_MATH})"),
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
                        ms=MS(MODEL_MARKER[m], 6.2), color=MODEL_COLOUR[m], mec="white",
                        mew=0.7, ls="", zorder=4)
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
    # Natural order: invert_xaxis above already puts high liberal share on the
    # left. Reversing the labels as well flipped it back, so a model at 100%
    # liberal share was sitting under a tick reading "0%".
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Liberal share of stance (among opinionated responses)", fontsize=11)
    ax.text(0.0, 1.006, _style.ARROW_L + "  more liberal", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=9, color="#888888")
    ax.text(1.0, 1.006, "more conservative  " + _style.ARROW_R, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color="#888888")
    ax.text(1.045, 1.006, "neutral", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="#999999")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, mfc="white", mec=CES_COLOUR,
                          mew=1.8, label="US public (CES 2025)")]
    handles += [plt.Line2D([], [], marker=MODEL_MARKER[m], ls="",
                           ms=MS(MODEL_MARKER[m], 6.6), color=MODEL_COLOUR[m],
                           mec="white", mew=0.7, label=MODEL_LABEL[m])
                for m in MODELS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.135),
              ncol=6, frameon=False, fontsize=9.5, columnspacing=1.3, handletextpad=0.4)
    fig.subplots_adjust(left=0.235, right=0.90, top=0.95, bottom=0.11)

    for ext in fmts:
        fig.savefig(out / f"fig_ces_levels.{ext}", bbox_inches="tight")
    plt.close(fig)


PARTY = {"democrat", "independent", "republican"}

# Panel order for the party-level figure: most to least liberal target.
# Panel titles name the party plainly; the caption says these are the party-label
# cues, so quoting the full system prompt in every title only crowds the panel.
PARTY_ORDER = [("democrat", "Democrat"),
               ("independent", "Independent"),
               ("republican", "Republican")]


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Unlike the normal (Wald) interval it
    cannot leave [0, 1] and does not degenerate to zero width at p = 0 or 1, the two
    failure modes that matter here (Brown, Cai & DasGupta 2001)."""
    if n < 1:
        return np.nan, np.nan
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    halfw = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    # clamp: at p = 0 or 1 the closed form lands on the boundary to within float
    # epsilon (~1e-16), which is harmless on the plot but noisy in diagnostics.
    return (float(np.clip((centre - halfw) / denom, 0.0, 1.0)),
            float(np.clip((centre + halfw) / denom, 0.0, 1.0)))


def _party_issue_support(results_dir: Path, issues_csv: Path):
    """Per issue x party-cue x model share of SUPPORT for the proposition, with a
    95% CI clustered on writing template.

    Why support and not the liberal score: the liberal/conservative axis inverts
    against the row label on 8 of the 19 issues (supporting the border wall is the
    CONSERVATIVE position, so a strongly liberal score there means strongly
    AGAINST the wall). Plotting the liberal score therefore reads backwards on
    those rows. ``liberal_sign`` records which side is the support side, so
    ``score == liberal_sign`` is the support indicator on every issue and the axis
    means the same thing on every row.

    Restricted to responses that took a side (score != 0), which is what makes it
    comparable to the CES forced choice.

    Estimand and interval are both at the TEMPLATE grain: ``support`` is the mean
    of the per-template support rates (templates weighted equally, not by how many
    of their generations happened to commit), and the interval is Wilson on the
    number of templates. Two reasons not to use a normal SE on the response count.
    First, the earlier SE -- the spread of per-template means -- collapses to
    exactly zero in 79 of 285 cells, because when every template agrees there is no
    spread to measure; the figure then asserted p = 1.000 with no uncertainty off as
    few as 21 responses, and a template bootstrap cannot repair that either. Wilson
    keeps continuity mass at the boundary and stays inside [0, 1] by construction
    (the old interval left it in 63 cells). Second, using the response count would
    treat the three generations per template as independent draws, which runs ~20%
    narrower than a template-clustered bootstrap on the 3-rep open models.

    ``commit`` (= n_dir / n_all) carries the denominator that conditioning hides.
    A share of 0.80 means something very different on 41 of 51 committed responses
    than on 41 of 51 responses total, and the caller drops low-commit cells outright
    so a hedged minority cannot be misread as a confident position.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_thesis_figures as _M

    sign = _M.load_liberal_sign(issues_csv)
    data = _M.load(results_dir)
    rows = []
    for grp, _ in PARTY_ORDER:
        for m in MODELS:
            df = data[m]
            sub = df[_M.cue_mask(df, "explicit_political", grp)]
            for iss, chunk in sub.groupby("issue_id"):
                if iss not in sign:
                    continue
                dd = chunk[chunk[_M.SCORE] != 0]
                if len(dd) < 2:
                    continue
                supports = (dd[_M.SCORE] == sign[iss]).astype(float)
                per_tmpl = supports.groupby(dd["tmpl"]).mean()
                n_t = len(per_tmpl)
                p = float(per_tmpl.mean())
                lo, hi = _wilson(p, n_t)
                rows.append({"cue_group": grp, "model": m, "ces_variable": iss,
                             "support": p, "lo": lo, "hi": hi,
                             "raw_share": float(supports.mean()),
                             "n_tmpl": n_t, "n_dir": len(dd), "n_all": len(chunk),
                             "commit": len(dd) / len(chunk)})
    return pd.DataFrame(rows), sign


def fig_party_levels(detail_csv: Path, results_dir: Path, issues_csv: Path,
                     out: Path, fmts) -> None:
    """Support for each proposition under each party cue, model vs. that subgroup's
    real CES support — the view the shift dumbbell cannot show.

    The dumbbell compares *shifts*, which assumes the model starts where the US
    public starts; it does not (baselines sit far more liberal). Worse, the per-cue
    average of the absolute gap cancels: Qwen's mean gap under the Republican cue
    is near zero, which reads as perfect calibration, but that is a large overshoot
    on guns/climate netting against a large undershoot on immigration. Only the
    issue grain shows it, hence one row per issue.

    Both sides are the share supporting the proposition among those who took a
    side, so a row reads directly: "83% of Democrats support this, the models only
    12%".

    Cells where fewer than COMMIT_MIN of responses took a side are omitted rather
    than flagged: a conditional share resting on a hedged minority is not a position,
    and drawing it (even hollow) invites the reader to place it anyway. A missing
    marker in a row is therefore itself informative — that model mostly refused the
    proposition under that cue.
    """
    d = pd.read_csv(detail_csv)
    d = d[(d.cue_family == "explicit_political") & (d.model.isin(MODELS))]
    lvl, sign = _party_issue_support(results_dir, issues_csv)

    # CES side: ces_lib_share is the LIBERAL share, so flip it on the issues where
    # supporting the proposition is the conservative position. Keyed on
    # (cue_group, issue) -- the anchor is the SUBGROUP's own position and differs
    # per panel (Republicans support the border wall 89%, Democrats 17%), so
    # collapsing on issue alone silently reuses one subgroup in all three panels.
    lib = d.groupby(["cue_group", "ces_variable"]).ces_lib_share.first()
    ces = {k: (v if sign.get(k[1], 1) > 0 else 1 - v) for k, v in lib.items()}
    # Sampling uncertainty on the target: Wilson on the subgroup's own n (4.5k-5.6k
    # per party), the same interval used for the model side so the two are read on
    # like terms. It ignores CES survey weights, so it is a floor on the true
    # sampling error (a design effect would widen it); at n ~ 5k the half-width is
    # <= 1.4pp either way, far below the model-vs-target gaps this figure is about.
    ces_n = d.groupby(["cue_group", "ces_variable"]).ces_n.first()
    ces_ci = {k: _wilson(ces[k], int(ces_n[k])) for k in ces}
    name = d.groupby("ces_variable").issue.first()
    order = (d.groupby("ces_variable").dem_rep_gap.first()
             .sort_values(ascending=False).index.tolist())
    yof = {iss: i for i, iss in enumerate(order)}

    fig, axes = plt.subplots(1, len(PARTY_ORDER), figsize=(14.6, 7.3),
                             gridspec_kw={"wspace": 0.05}, squeeze=False)
    n_drop = 0
    # One line per issue: no per-model dodge, so a row is a single strip of the
    # support axis and the reader compares positions along it (the models cluster,
    # which is the point). Whiskers are therefore co-linear and can overlap; drawing
    # them widest-first keeps the tighter, more informative ones on top.

    for j, (grp, title) in enumerate(PARTY_ORDER):
        ax = axes[0][j]
        sub = lvl[lvl.cue_group == grp]
        for i in range(len(order)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#5A5A66", alpha=0.045, zorder=0)
        # Quarter gridlines to read levels off; the even split is the only one that
        # carries meaning (for vs. against), so it alone gets weight.
        for gx in (0.25, 0.75):
            ax.axvline(gx, color="#D6D6DB", lw=0.7, zorder=1)
        ax.axvline(0.5, color="#9A9AA2", lw=1.0, zorder=1)

        for iss in order:
            yy = yof[iss]
            rows = sub[sub.ces_variable == iss]
            if not len(rows):
                continue
            c = float(ces[(grp, iss)])
            cells = []
            for m in MODELS:
                r = rows[rows.model == m]
                if not len(r):
                    continue
                # Drop cells resting on a hedged minority (see docstring).
                if float(r.commit.iloc[0]) < COMMIT_MIN:
                    n_drop += 1
                    continue
                cells.append((m, float(r.support.iloc[0]),
                              float(r.lo.iloc[0]), float(r.hi.iloc[0])))
            for m, x, lo, hi in sorted(cells, key=lambda t: -(t[3] - t[2])):
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [yy, yy], color=MODEL_COLOUR[m], lw=2.4,
                            alpha=0.30, zorder=3, solid_capstyle="round")
            for m, x, lo, hi in cells:
                ax.plot(x, yy, marker=MODEL_MARKER[m], ms=MS(MODEL_MARKER[m], 6.0),
                        mfc=MODEL_COLOUR[m], mec="white", mew=0.8, ls="", zorder=4)
            # The target rides the same line as the models: a dark X with a white
            # halo (so it stays legible when a model lands on top of it) and its
            # Wilson interval as a short rule through it. That interval is usually
            # narrower than the X, which is the honest read: at n ~ 5k the survey
            # target is precise and every visible gap is the model's.
            clo, chi = ces_ci[(grp, iss)]
            if np.isfinite(clo) and np.isfinite(chi):
                ax.plot([clo, chi], [yy, yy], color=CES_COLOUR, lw=2.4, alpha=0.55,
                        zorder=5, solid_capstyle="round")
            ax.plot(c, yy, marker="X", ms=MS("X", 6.6), mfc=CES_COLOUR,
                    mec=CES_COLOUR, mew=0.5, ls="", zorder=6, path_effects=XHALO)

        # Left-to-right = less to more support. No inversion: the axis means the
        # same thing on every row now, so it needs no mirroring.
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(order) - 0.6, -0.6)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9,
                           color="#555555")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, color="#BBBBBB")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#BBBBBB")
        ax.text(0.5, 1.035, title, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11.5, color="#222222")
        # Hairline under each column header, so the three panels read as columns of
        # one table rather than three loose scatters.
        ax.plot([0.0, 1.0], [1.025, 1.025], transform=ax.transAxes,
                color="#C9C9CE", lw=0.8, clip_on=False, zorder=1)
        if j == 0:
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([_tw.fill(name[iss], 30, break_long_words=False)
                                for iss in order], fontsize=8.8, linespacing=1.05,
                               color="#222222")
        else:
            ax.set_yticks([])

    fig.supxlabel("Share supporting the proposition "
                  "(of responses that took a side)", fontsize=10.5, y=0.028,
                  color="#333333")
    handles = [plt.Line2D([], [], marker="X", ls="", ms=MS("X", 7.0),
                          color=CES_COLOUR, label="CES subgroup (target)")]
    handles += [plt.Line2D([], [], marker=MODEL_MARKER[m], ls="",
                           ms=MS(MODEL_MARKER[m], 6.4), color=MODEL_COLOUR[m],
                           mec="white", mew=0.7, label=MODEL_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               ncol=6, frameon=False, fontsize=9.5, columnspacing=1.3,
               handletextpad=0.4, labelcolor="#333333")
    # The drop rule (commit < COMMIT_MIN) is explained in the LaTeX caption rather
    # than on the figure, per the no-on-figure-annotation style.
    print(f"  fig_ces_party_levels: dropped {n_drop} hedged model x issue x cue "
          f"cells (commit < {COMMIT_MIN:.2f})")
    fig.subplots_adjust(left=0.155, right=0.99, top=0.945, bottom=0.10)

    for ext in fmts:
        fig.savefig(out / f"fig_ces_party_levels.{ext}", bbox_inches="tight")
    plt.close(fig)


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
         r"shifts flatten the corresponding CES subgroup differences, and "
         r"$\beta>1$ that they exaggerate them. The "
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
    ap.add_argument("--results-dir", default="results/full_3x")
    ap.add_argument("--issues-csv", default="data/input/issues_experiment.csv")
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
    fig_party_levels(Path(args.detail), Path(args.results_dir),
                     Path(args.issues_csv), fd, fmts)
    write_slope_table(load_slopes(Path(args.slopes)), Path(args.shift_table), fd)
    print(f"\nWrote fig_ces_dumbbell + fig_ces_levels + fig_ces_party_levels "
          f"+ calibration_slope_table.{{md,tex}} to {fd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
