#!/usr/bin/env python3
"""Three headline thesis figures for the cue-steering run (3 open-source models).

Classifier of record: DeBERTa ``bert_liberal_score`` in {-1, 0, +1}
(+1 = wrote the liberal side, -1 = conservative side, 0 = neutral). Titles are
omitted deliberately (a LaTeX caption carries the takeaway).

  fig1_forest.png        Two-panel forest over the four cue types (party label,
                         state, race x gender, name; ordered by how directly they
                         signal politics), one marker + 95% CI (clustered on CES
                         issue) per model. LEFT = absolute stance (mean liberal
                         score, with each model's no-cue baseline as a dashed
                         reference); RIGHT = shift vs. that baseline. The shift
                         axis is held at +/-SHIFT_EDGE so the small cue effects
                         stay readable; the few off-scale points (the Republican
                         cue on the larger models) are clamped to the edge and
                         drawn as stars, with their true values reported in the
                         caption/text.
  fig2_calibration.png   Model stance shift (cued - baseline) vs. the REAL CES
                         2025 subgroup shift (subgroup - population). y = x is
                         perfect calibration; steeper than the line = exaggerates
                         the real gap, flatter (toward y = 0) = flattens it.
                         Marker shape = model, colour = cue family.
  fig3_composition.png   Baseline (no-cue) Liberal / Neutral / Conservative
                         response mix (liberal on the LEFT, conservative on the
                         RIGHT), one row per issue, one column per model, as
                         stacked bars. Rows ordered by cross-model liberal lean
                         (this is the ceiling/floor context for Fig 1). A coloured
                         thumbs-up on each row marks the "support/pro" side of that
                         issue: blue = the liberal side supports it, red = the
                         conservative side supports it.

Reads results/full_3x/bert_eval_<model>.csv (slim per-model stance scores) and
results/full_3x/ces_estimates.csv (weighted CES 2025 estimates from the .dta).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

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

# 3 open-source (3-rep) models + 2 frontier (1-rep, API) models. Order = legend order.
MODELS = ["qwen", "gemma", "llama", "gpt56terra", "sonnet5"]
MODEL_LABEL = {"qwen": "Qwen-3.6-27B", "gemma": "Gemma-3-12B", "llama": "Llama-3.1-8B",
               "gpt56terra": "GPT-5.6 Terra", "sonnet5": "Claude Sonnet 5"}
# Okabe-Ito orange/green/blue: the CB-safe trio that stays separable under
# deuteranopia/protanopia (-> yellow / grey / blue). Avoid the orange/green/pink
# set, where green and reddish-purple both desaturate to the same grey.
# Frontier pair: reddish-purple + sky-blue (both CB-safe, distinct from the trio).
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#0072B2",
                "gpt56terra": "#CC79A7", "sonnet5": "#56B4E9"}
# Redundant per-model marker shape so identity survives even total colour loss.
MODEL_MARKER = {"qwen": "o", "gemma": "s", "llama": "^", "gpt56terra": "D", "sonnet5": "v"}

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _common import EVAL_PREFIX, SCORE_COL  # classifier-of-record switch (SCORER env)

SCORE = SCORE_COL

# --- company logos as point markers + legend keys ---------------------------
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
import matplotlib.image as _mpimg
import os as _os

# Thesis = clean shape markers (Okabe-Ito). Public-facing = company logos.
# Toggle with THESIS_LOGOS=1; the logo build writes to its own figures dir.
USE_LOGOS = _os.environ.get("THESIS_LOGOS", "0") == "1"

_LOGO_DIR = Path(__file__).resolve().parents[2] / "figures" / "logos" / "png"
_LOGO_CACHE: dict = {}


def _logo(model: str):
    if model not in _LOGO_CACHE:
        _LOGO_CACHE[model] = _mpimg.imread(_LOGO_DIR / f"{model}.png")
    return _LOGO_CACHE[model]


def place_logo(ax, x, y, model, target_px=14, zorder=5):
    """Drop a company logo at (x, y) in data coords, sized to ~target_px tall."""
    oi = OffsetImage(_logo(model), zoom=target_px / _logo(model).shape[0])
    ax.add_artist(AnnotationBbox(oi, (x, y), frameon=False, pad=0.0, zorder=zorder,
                                 box_alignment=(0.5, 0.5), clip_on=False))


class _HandlerLogo(HandlerBase):
    """Legend handler that renders a model's logo instead of a marker glyph."""
    def __init__(self, model, target_px=16, **kw):
        super().__init__(**kw)
        self.model, self.target_px = model, target_px

    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        oi = OffsetImage(_logo(self.model), zoom=self.target_px / _logo(self.model).shape[0])
        ab = AnnotationBbox(oi, (width / 2.0 - xd, height / 2.0 - yd), frameon=False,
                            pad=0.0, box_alignment=(0.5, 0.5))
        ab.set_transform(trans)
        return [ab]


def logo_legend(ax, models, extra_handles=None, target_px=16, **legend_kw):
    """A legend whose model rows are company logos (points-and-legend request)."""
    handles = [plt.Line2D([], [], ls="", label=MODEL_LABEL[m]) for m in models]
    hmap = {h: _HandlerLogo(m, target_px) for h, m in zip(handles, models)}
    if extra_handles:
        handles += extra_handles
    return ax.legend(handles=handles, handler_map=hmap, **legend_kw)


def logo_key(ax, models, x=0.75, y=0.18, dy=0.030, target_px=14):
    """Draw a compact logo-and-name key in axes coordinates.

    AnnotationBbox images inside a normal Matplotlib legend are not preserved by
    every backend.  Placing this key directly on the axes keeps the raster logos
    visible in both the PNG and vector-PDF outputs.
    """
    for i, model in enumerate(models):
        yy = y - i * dy
        oi = OffsetImage(_logo(model), zoom=target_px / _logo(model).shape[0])
        ax.add_artist(AnnotationBbox(
            oi, (x, yy), xycoords=ax.transAxes, frameon=False, pad=0.0,
            box_alignment=(0.5, 0.5), zorder=8, clip_on=False))
        ax.text(x + 0.035, yy, MODEL_LABEL[model], transform=ax.transAxes,
                ha="left", va="center", fontsize=9, color="#111111", zorder=8)


class _HandlerLogoShape(HandlerBase):
    """Legend key = the model's marker shape next to its logo (for figures where the
    point encodes model-by-shape but we still want the logo in the legend)."""
    def __init__(self, model, marker, target_px=14, **kw):
        super().__init__(**kw)
        self.model, self.marker, self.target_px = model, marker, target_px

    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        ln = plt.Line2D([width * 0.30 - xd], [height / 2.0 - yd], marker=self.marker,
                        ls="", color="#444444", mec="white", mew=0.5, markersize=7)
        ln.set_transform(trans)
        oi = OffsetImage(_logo(self.model), zoom=self.target_px / _logo(self.model).shape[0])
        oi.set_offset((width * 0.74 - xd, height / 2.0 - yd))
        oi.set_transform(trans)
        return [ln, oi]


def logo_shape_legend(ax, models, markers, target_px=14, **legend_kw):
    handles = [plt.Line2D([], [], ls="", label=MODEL_LABEL[m]) for m in models]
    hmap = {h: _HandlerLogoShape(m, markers[m], target_px) for h, m in zip(handles, models)}
    return ax.legend(handles=handles, handler_map=hmap, **legend_kw)


def _logo_point(ax, x, xerr, edge, y, colour, model):
    """Logo marker + CI if on-scale; off-scale falls back to a colored star at the
    edge (keeps model colour + the 'clamped' signal, which a logo can't convey)."""
    if abs(x) <= edge:
        if xerr is not None:
            ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor=colour,
                        elinewidth=1.3, capsize=0, zorder=3)
        place_logo(ax, x, y, model)
    else:
        ax.plot(np.copysign(edge, x), y, marker="*", ms=13, color=colour,
                mec="white", mew=0.7, zorder=4, clip_on=False)


def _tcrit(n: int) -> float:
    """Two-sided 95% critical value on n-1 df.

    NOT 1.96. These CIs are clustered on CES issue, so the SD is estimated from
    only n = 19 issue-level values, and (est - truth)/SE follows Student's t on
    18 df, not the normal. Using 1.96 makes every interval 7% too narrow: a
    nominal 95% interval covers 93.4%, i.e. a 6.6% false-positive rate. The
    degrees of freedom come from the number of CLUSTERS, not the ~8k generations
    per cell -- clustering buys robustness to within-issue correlation and pays
    for it in df."""
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))

# Composition palette
C_CON, C_NEU, C_LIB = "#B2182B", "#DBDBDB", "#3A6EA5"


def _thumb_marker():
    """A thumbs-up silhouette as a single closed Path, usable as a matplotlib
    marker. Scales uniformly with markersize and renders as vector in the PDF
    (unlike the emoji glyph, which lives only in a colour-bitmap font)."""
    from matplotlib.path import Path as _P
    v = np.array([
        (0.02, 0.02), (0.02, 0.56), (0.12, 0.70), (0.12, 0.88),
        (0.19, 0.98), (0.30, 0.99), (0.37, 0.90), (0.35, 0.66),
        (0.33, 0.58), (0.82, 0.58), (0.95, 0.50), (0.95, 0.40),
        (0.88, 0.30), (0.93, 0.18), (0.84, 0.06), (0.86, 0.02),
        (0.30, 0.02), (0.02, 0.02),
    ], float)
    v -= v.mean(0)
    v /= np.abs(v).max()
    return _P(v, closed=True)


THUMB = _thumb_marker()


def _save(fig, out: Path, stem: str, fmts) -> None:
    """Write a figure in every requested format. PDF is vector (text stays
    selectable and scales without pixelation) and is the right choice for LaTeX;
    PNG is kept for quick previewing."""
    for ext in fmts:
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data + cue masks
# --------------------------------------------------------------------------- #
def load(results_dir: Path) -> dict[str, pd.DataFrame]:
    cols = ["prompt_id", "arm", "cue_condition", "cue_family", "cue_group", "issue_id", SCORE]
    out = {}
    for m in MODELS:
        df = pd.read_csv(results_dir / f"{EVAL_PREFIX}_{m}.csv", usecols=cols, low_memory=False)
        # prompt_id is "<issue>__t<k>__<condition>__r<rep>": pull the template id,
        # needed to match each cue against a baseline on the SAME prompts.
        df["tmpl"] = df.prompt_id.str.split("__", expand=True)[1]
        out[m] = df
    return out


def base_mask(df: pd.DataFrame):
    """Full no-cue baseline (all 145 templates). Correct for arm-A cues, which are
    run on all 145; use ``baseline_for`` when contrasting an arm-B cue."""
    return (df.arm == "A") & (df.cue_condition == "baseline")


def baseline_for(df: pd.DataFrame, cued_mask):
    """Baseline restricted to the templates the cue itself was run on.

    Arm B (state / name cues) runs on a 35-template SUBSET of the 145 baseline
    templates, and that subset is not a random draw: its baseline mean is ~0.039
    more liberal than the full bank. Contrasting an arm-B cue against the full
    baseline therefore charges that template-composition difference to the cue --
    an artefact of the same magnitude as the effects being estimated (it was
    manufacturing a uniform "every implicit cue nudges Llama liberal" pattern).
    Matching on template removes it. Arm A already spans all 145, so this is a
    no-op there. Mirrors the estimator in analysis/lib/_common.py."""
    bm = base_mask(df)
    tmpl = df.loc[cued_mask, "tmpl"].unique()
    if len(tmpl) < df.loc[bm, "tmpl"].nunique():
        bm = bm & df.tmpl.isin(tmpl)
    return bm


def cue_mask(df: pd.DataFrame, family: str, group: str):
    if family.startswith("explicit"):
        return (df.arm == "A") & (df.cue_condition == f"{family}_{group}")
    # implicit cues live in arm B, identified by family + group (cue_condition
    # carries the per-instance name/state suffix, so we don't match on it).
    return (df.arm == "B") & (df.cue_family == family) & (df.cue_group == group)


# --- estimators: 95% CI clustered on issue -------------------------------- #
def _issue_means(df, mask):
    return df[mask].groupby("issue_id")[SCORE].mean()


def shift_ci(df, cued_mask, bmask):
    d = (_issue_means(df, cued_mask) - _issue_means(df, bmask)).dropna()
    pooled = df[cued_mask][SCORE].mean() - df[bmask][SCORE].mean()
    return pooled, _tcrit(len(d)) * d.std(ddof=1) / np.sqrt(len(d))


def level_ci(df, mask):
    """Absolute mean liberal score with a 95% CI clustered on issue (SE from the
    spread of per-issue means). This is the alignment: where a cue actually lands
    on the -1..+1 scale, which sets how much headroom is left to move."""
    m = _issue_means(df, mask)
    return df[mask][SCORE].mean(), _tcrit(len(m)) * m.std(ddof=1) / np.sqrt(len(m))


# Four cue types, ordered by how directly they signal the user's politics
# (most direct -> least). The family/group keys index the data and are unchanged;
# only the display labels differ. (band label, family, group, row label)
FOREST_BANDS = [
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
        ("implicit_demographic", "black_woman", "Name: Black woman"),
        ("implicit_demographic", "black_man", "Name: Black man"),
        ("implicit_demographic", "white_woman", "Name: White woman"),
        ("implicit_demographic", "white_man", "Name: White man")]),
]


# --------------------------------------------------------------------------- #
# Fig 1 — forest
# --------------------------------------------------------------------------- #
def _layout_rows(bands_spec):
    """Lay bands out top -> bottom with a gap between them. Returns the row list
    (y, family, group, label), band spans, and the y extent."""
    rows, bands, y = [], [], 0.0
    for band, items in bands_spec:
        y0 = y
        for fam, grp, lbl in items:
            rows.append((y, fam, grp, lbl))
            y += 1.0
        bands.append((band, (y0 + y - 1.0) / 2.0, y0 - 0.5, y - 0.5))
        y += 0.9  # gap
    return rows, bands, y


# The SHIFT axis is held short so the many small cue effects stay readable; any
# point beyond it is clamped to the edge and drawn as a star (true value given in
# the text). In practice only the Republican shift on the larger models exceeds
# it. The STANCE axis instead runs the full -1..+1: per-issue means do reach the
# ends, and trimming the scale piled them into a fake stack at the edge.
STANCE_EDGE = 1.0
SHIFT_EDGE = 0.30


def _forest_point(ax, x, xerr, edge, y, colour, marker):
    """Plot one marker: a shape + CI if on-scale, else a star clamped to the edge.

    ``xerr=None`` draws the marker bare. The left (absolute stance) panel uses
    this deliberately: see ``_draw_forest_block``."""
    if abs(x) <= edge:
        ax.errorbar(x, y, xerr=xerr, fmt=marker, ms=6.5, color=colour, ecolor=colour,
                    elinewidth=1.3, capsize=0, zorder=3, mec="white", mew=0.7)
    else:  # off-scale: clamp to the edge, mark with a star, drop the CI
        ax.plot(np.copysign(edge, x), y, marker="*", ms=13, color=colour,
                mec="white", mew=0.7, zorder=4, clip_on=False)


def _model_point(ax, x, xerr, edge, y, colour, m):
    """One model's point: a company logo (public build) or a shape (thesis)."""
    if USE_LOGOS:
        _logo_point(ax, x, xerr, edge, y, colour, m)
    else:
        _forest_point(ax, x, xerr, edge, y, colour, MODEL_MARKER[m])


def _model_legend(ax, models, extra_handles=None, **legend_kw):
    if USE_LOGOS:
        return logo_legend(ax, models, extra_handles=extra_handles, **legend_kw)
    handles = [plt.Line2D([], [], marker=MODEL_MARKER[m], ls="", color=MODEL_COLOUR[m],
                          label=MODEL_LABEL[m], mec="white", mew=0.7) for m in models]
    if extra_handles:
        handles += extra_handles
    return ax.legend(handles=handles, **legend_kw)


def _frame_rows(ax, rows, bands, ylabels: bool):
    """Shared row scaffolding: cue-group bands, row ticks, and orientation."""
    # Direction (liberal left / conservative right) is carried by the axis-edge
    # labels; no blue/red tint (it hurt readability). Keep only the faint
    # alternating band shading that delimits the four cue-type groups.
    for i, (_, _, lo, hi) in enumerate(bands):
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#000000", alpha=0.04, zorder=0)
    ax.set_yticks([r[0] for r in rows])
    ax.set_ylim(max(r[0] for r in rows) + 0.7, min(r[0] for r in rows) - 0.7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.invert_xaxis()  # liberal on the LEFT, conservative on the RIGHT (matches Fig 3)
    if ylabels:
        ax.set_yticklabels([r[3] for r in rows], fontsize=10, rotation=28,
                           ha="right", va="center", rotation_mode="anchor")
    else:
        ax.tick_params(axis="y", labelleft=False)


def _direction_tags(ax, left, right):
    ax.text(0.0, 1.004, left, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="#888888")
    ax.text(1.0, 1.004, right, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#888888")


# The two panels answer different questions and are deliberately drawn in
# different idioms, because they were being conflated. Both could carry a "95% CI
# clustered on issue", but they differ ~5-9x in width: the level SE is driven by
# how far apart the 19 CES issues sit (~0.15 for every cue, baseline included --
# it measures the issue set, not the cue), whereas the shift is a paired
# within-issue contrast, so the issue main effect cancels and what remains is
# heterogeneity in *cue response*. Showing both invited two misreadings: wide
# level bars read as a cue's uncertainty, and overlapping level bars (strongly
# correlated across cues -- same issues, same model) read as "no difference".
# So: STANCE is descriptive and shows its distribution; SHIFT is inferential and
# is the only panel carrying CIs.
def _draw_stance_panel(ax, data, base_lvl, offs, rows, bands, ylabels=True, legend=True):
    """Descriptive panel: where each cue lands on the -1..+1 stance scale."""
    for yy, fam, grp, _ in rows:
        for m, off in zip(MODELS, offs):
            df = data[m]
            cm = cue_mask(df, fam, grp)
            # The 19 per-issue means themselves, as a strip behind the mean marker.
            # A dot cloud reads as a distribution, not an estimate -- unlike a bar,
            # which is what made the old level CI so easy to misread. It also shows
            # the issue spread honestly instead of compressing it into a width.
            im = _issue_means(df, cm)
            ax.plot(np.clip(im.values, -STANCE_EDGE, STANCE_EDGE),
                    np.full(len(im), yy + off), ls="", marker="o", ms=2.6,
                    color=MODEL_COLOUR[m], alpha=0.30, mec="none", zorder=2)
            lv, _ = level_ci(df, cm)
            _model_point(ax, lv, None, STANCE_EDGE, yy + off, MODEL_COLOUR[m], m)

    # Dashed per-model line = that model's own no-cue anchor (where it starts).
    # Contrast the shift panel's solid zero, which is a null.
    #
    # Drawn per BAND, not across the whole panel: arm-B bands (state, name) run on
    # a 35-template subset whose no-cue mean is ~0.039 more liberal than the full
    # bank, so those rows must be read against THEIR baseline. A single line would
    # reintroduce exactly the composition confound baseline_for() removes.
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    for (_, items), (_, _, lo, hi) in zip(FOREST_BANDS, bands):
        fam0, grp0 = items[0][0], items[0][1]
        for m in MODELS:
            df = data[m]
            bm = baseline_for(df, cue_mask(df, fam0, grp0))
            ax.plot([df[bm][SCORE].mean()] * 2, [lo, hi], color=MODEL_COLOUR[m],
                    lw=1.1, ls="--", alpha=0.8, zorder=2)
    ax.set_xlim(-STANCE_EDGE, STANCE_EDGE)
    _frame_rows(ax, rows, bands, ylabels)
    _direction_tags(ax, "←  liberal", "conservative  →")
    # No CI is named here: the cheapest signal that this panel makes no claim.
    ax.set_xlabel(r"Model stance  ($\bar{Y}_k$)", fontsize=11.5)
    if legend:
        extra = [
            plt.Line2D([], [], marker="o", ls="", ms=3.4, color="#777777", alpha=0.45,
                       label="per-issue mean (19 issues)"),
            plt.Line2D([], [], color="#999999", ls="--", lw=1.1,
                       label=r"model baseline ($\bar{Y}_{\mathrm{baseline}}$)")]
        _model_legend(ax, MODELS, extra_handles=extra, loc="lower right", frameon=False,
                      fontsize=9, labelspacing=0.7)


def _draw_shift_panel(ax, data, offs, rows, bands, ylabels=True, legend=True,
                      logo_points=False, monochrome=False):
    """Inferential panel: estimated cue effect, with 95% CIs clustered on issue."""
    for yy, fam, grp, _ in rows:
        for m, off in zip(MODELS, offs):
            df = data[m]
            cm = cue_mask(df, fam, grp)
            sh, se = shift_ci(df, cm, baseline_for(df, cm))
            colour = "#555555" if monochrome else MODEL_COLOUR[m]
            if logo_points:
                # Keep the model logo even when the estimate is outside the
                # displayed range.  A small outward arrow marks truncation.
                plotted = np.clip(sh, -SHIFT_EDGE, SHIFT_EDGE)
                if abs(sh) <= SHIFT_EDGE:
                    ax.errorbar(sh, yy + off, xerr=se, fmt="none", ecolor=colour,
                                elinewidth=1.3, capsize=0, zorder=3)
                else:
                    # A small asterisk beside the clipped logo marks an estimate
                    # beyond the displayed range without overwhelming the logo.
                    ax.annotate("*", (plotted, yy + off),
                                xytext=(8 if sh < 0 else -8, 0),
                                textcoords="offset points", ha="center", va="center",
                                fontsize=12, fontweight="bold", color=colour,
                                zorder=6, clip_on=False)
                place_logo(ax, plotted, yy + off, m, target_px=10.5, zorder=5)
            else:
                _model_point(ax, sh, se, SHIFT_EDGE, yy + off, colour, m)

    ax.axvline(0, color="#222222", lw=1.5, zorder=2)  # the null
    ax.set_xlim(-SHIFT_EDGE * 1.06, SHIFT_EDGE * 1.06)
    _frame_rows(ax, rows, bands, ylabels)
    _direction_tags(ax, "←  more liberal", "more conservative  →")
    ax.set_xlabel(r"Shift vs. no-cue baseline  ($\hat{\Delta}_k$), 95% CI", fontsize=11.5)
    if legend:
        off_marker = "$*$" if logo_points else "*"
        extra = [plt.Line2D([], [], marker=off_marker, ls="", ms=11, color="#777777",
                            mec="white", mew=0.7,
                            label=f"off scale: $|\\hat{{\\Delta}}_k| > {SHIFT_EDGE:g}$")]
        if logo_points:
            logo_key(ax, MODELS)
            ax.plot(0.75, 0.025, marker=off_marker, ms=11, color="#777777",
                    mec="white", mew=0.7, ls="", transform=ax.transAxes,
                    clip_on=False, zorder=8)
            ax.text(0.785, 0.025,
                    f"off scale: $|\\hat{{\\Delta}}_k| > {SHIFT_EDGE:g}$",
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=9, color="#111111", zorder=8)
        else:
            _model_legend(ax, MODELS, extra_handles=extra, loc="lower right",
                          frameon=False, fontsize=9, labelspacing=0.7)


def _panel_header(ax, title, subtitle):
    ax.text(0.5, 1.055, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#555555")
    # Second line spells out the split the bare/CI contrast already encodes.
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, style="italic", color="#999999")


def _forest_setup(data):
    rows, bands, _ = _layout_rows(FOREST_BANDS)
    offs = np.linspace(0.24, -0.24, len(MODELS))  # model jitter within a row
    base_lvl = {m: level_ci(data[m], base_mask(data[m]))[0] for m in MODELS}
    return rows, bands, offs, base_lvl


def fig_forest(data, out: Path, fmts):
    """Combined two-panel version (kept for slides/talks; the thesis uses the two
    standalone panels below as LaTeX subfigures, so each gets its own caption)."""
    rows, bands, offs, base_lvl = _forest_setup(data)
    # NB: no sharey — both axes get identical ylim/yticks, so they stay aligned,
    # and sharing would let axR's blank labels clobber axL's.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 9.4),
                                   gridspec_kw={"wspace": 0.08})
    # In the combined figure the headers do the descriptive/inferential signposting
    # that the separate subcaptions do when the panels stand alone.
    _draw_stance_panel(axL, data, base_lvl, offs, rows, bands, ylabels=True)
    _panel_header(axL, "ABSOLUTE STANCE", "where each cue lands")
    _draw_shift_panel(axR, data, offs, rows, bands, ylabels=False, legend=False)
    _panel_header(axR, "SHIFT vs. no-cue baseline", "the estimated cue effect")
    # Combined: the star legend alone on the right, models are labelled on the left.
    axR.legend(handles=[plt.Line2D([], [], marker="*", ls="", ms=12, color="#777777",
                                   mec="white", mew=0.7,
                                   label=f"$\\hat{{\\Delta}}_k < -{SHIFT_EDGE:g}$")],
               loc="lower right", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.08)
    _save(fig, out, "fig1_forest", fmts)


# Panel (a) carries the row labels; panel (b) repeats the same 14 rows in the same
# order, so its labels are dropped and the space goes to the data. That only works
# if the two land SIDE BY SIDE with their rows aligned, which needs the plot areas
# to be the same size on the page. Both figures are the same height and reserve the
# same axes width in inches; (b) is narrower by exactly the label gutter it drops.
# So they must be included at widths in the ratio PANEL_W_A : PANEL_W_B (see the
# LaTeX snippet in docs/), NOT at the same \textwidth fraction -- equal widths would
# stretch (b)'s rows taller than (a)'s and break the alignment.
PANEL_H = 9.4
PANEL_W_A = 7.6          # includes the row-label gutter
_AXES_W = PANEL_W_A * (0.98 - 0.21)   # shared plot-area width, inches
_B_LEFT = 0.035
PANEL_W_B = round(_AXES_W / (0.98 - _B_LEFT), 2)


def fig_forest_panels(data, out: Path, fmts):
    """The same two panels as standalone files, for \\subfigure (a) and (b). No
    in-figure header — the LaTeX subcaption states whether the panel is descriptive
    or inferential. Row labels appear once, on (a)."""
    rows, bands, offs, base_lvl = _forest_setup(data)

    figA, axA = plt.subplots(figsize=(PANEL_W_A, PANEL_H))
    _draw_stance_panel(axA, data, base_lvl, offs, rows, bands, ylabels=True)
    figA.subplots_adjust(left=0.21, right=0.98, top=0.96, bottom=0.08)
    _save(figA, out, "fig1a_stance", fmts)

    # The shift panel now stands alone in the thesis: restore its cue labels and
    # use logos with neutral confidence intervals instead of a colour/shape key.
    figB, axB = plt.subplots(figsize=(PANEL_W_A, PANEL_H))
    _draw_shift_panel(axB, data, offs, rows, bands, ylabels=True,
                      logo_points=True, monochrome=True)
    figB.subplots_adjust(left=0.21, right=0.98, top=0.96, bottom=0.08)
    _save(figB, out, "fig1b_shift", fmts)


# --------------------------------------------------------------------------- #
# Fig 2 — CES calibration scatter
# --------------------------------------------------------------------------- #
FAM_MARKER = {
    "explicit_political": "o",
    "explicit_demographic": "s",
    "implicit_political": "^",
    "implicit_demographic": "D",
}
# Fig 2 encodes MODEL by marker shape and CUE FAMILY by colour (Okabe-Ito, CB-safe).
FAM_COLOUR = {
    "explicit_political": "#D55E00",    # vermillion  — party labels
    "explicit_demographic": "#0072B2",  # blue        — race x gender
    "implicit_political": "#009E73",    # green       — location
    "implicit_demographic": "#CC79A7",  # purple      — name
}
FAM_MARKER_LABEL = {
    "explicit_political": "Party labels",
    "explicit_demographic": "Race × gender",
    "implicit_political": "Location",
    "implicit_demographic": "Name",
}


def _calibration_fits(robust_dir: Path):
    """Pooled Deming (errors-in-variables) calibration slope + intercept, fit with
    all cues and again with the three explicit party-identity cues removed. Same
    estimator as analysis/04_calibration/rq2_regression.py; shows how much of the
    calibration is carried by the party labels."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    from _regression import deming, x_var
    df = pd.read_csv(robust_dir / "model_shift_table.csv")
    party = {"democrat", "independent", "republican"}

    def fit(d):
        x, y = d["ces_shift_mean"].to_numpy(), d["model_shift"].to_numpy()
        delta = d["model_shift_var"].mean() / x_var(d).mean()
        return deming(x, y, delta)  # (slope, intercept)

    keep = ~((df.cue_family == "explicit_political") & (df.cue_group.isin(party)))
    return {"all": fit(df), "noparty": fit(df[keep])}


def fig_calibration(data, ces_table: Path, out: Path, fmts, robust_dir: Path):
    import math
    ces = pd.read_csv(ces_table)
    fits = _calibration_fits(robust_dir)
    fig, ax = plt.subplots(figsize=(9.0, 8.6))

    xs_all, ys_all = [], []
    for _, r in ces.iterrows():
        fam, grp = r.cue_family, r.cue_group
        x = r.ces_shift_mean
        xerr = [[x - r.ces_shift_ci_low], [r.ces_shift_ci_high - x]]  # CES 95% CI
        for m in MODELS:
            df = data[m]
            sh, se = shift_ci(df, cue_mask(df, fam, grp), base_mask(df))  # 95% half-width
            ax.errorbar(x, sh, xerr=xerr, yerr=se, fmt="none", ecolor=FAM_COLOUR[fam],
                        elinewidth=0.9, alpha=0.4, zorder=2, capsize=0)
            ax.scatter(x, sh, marker=MODEL_MARKER[m], s=70,
                       color=FAM_COLOUR[fam], edgecolor="white", linewidth=0.6,
                       zorder=3, alpha=0.9)
            xs_all.append(x); ys_all.append(sh)

    lim = max(0.3, np.nanmax(np.abs(xs_all + ys_all)) * 1.15)
    # "exaggerates" wedges: steeper than y=x (between the line and the y-axis),
    # i.e. the model's subgroup gap is bigger than the real CES gap.
    # Faint red "exaggerate" wedges (between y=x and the vertical axis). No hatch —
    # the fill is light enough that the dark/bold text labels stay legible on top.
    ax.fill_between([-lim, 0], [-lim, 0], [-lim, -lim], color="#B2182B", alpha=0.06, zorder=0)
    ax.fill_between([0, lim], [lim, lim], [0, lim], color="#B2182B", alpha=0.06, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color="#999999", ls="--", lw=1.2, zorder=1)
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)

    # Fitted calibration slopes: all cues vs. party-labels removed. The party
    # cues sit at the extreme x and carry most of the leverage, so dropping them
    # roughly halves the slope (0.75 -> 0.33).
    (b_all, a_all), (b_np, a_np) = fits["all"], fits["noparty"]
    xr = np.array([-lim, lim])
    # Both fits in black, distinguished by line style (solid vs dotted) rather than
    # hue, so they stay separable under colour-vision deficiency and never clash
    # with the model colours.
    ax.plot(xr, b_all * xr + a_all, color="#111111", lw=2.1, zorder=2)
    ax.plot(xr, b_np * xr + a_np, color="#111111", lw=2.3, ls=(0, (1, 1.1)), zorder=2)
    ax.text(0.42 * lim, b_all * 0.42 * lim + a_all + 0.008,
            rf"all cues:  $\beta={b_all:.2f}$", color="#111111", fontsize=9.5,
            ha="center", va="bottom", rotation=math.degrees(math.atan(b_all)),
            rotation_mode="anchor", fontweight="bold")
    ax.text(0.60 * lim, b_np * 0.60 * lim + a_np - 0.008,
            rf"party labels removed:  $\beta={b_np:.2f}$", color="#111111", fontsize=9.5,
            ha="center", va="top", rotation=math.degrees(math.atan(b_np)),
            rotation_mode="anchor", fontweight="bold")

    ax.text(-0.52 * lim, -0.86 * lim, "EXAGGERATES", color="#111111",
            fontsize=10, fontweight="bold", ha="center", va="center")
    ax.text(0.62 * lim, 0.02 * lim, "FLATTENS", color="#111111", fontsize=10,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.60 * lim, 0.72 * lim, "perfect calibration\n(model shift = real shift)",
            color="#333333", fontsize=9, ha="center", va="center", rotation=45,
            rotation_mode="anchor")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Real opinion shift in CES 2025  ($\mu_k^{CES}-\mu_{pop}^{CES}$)",
                  fontsize=11.5)
    ax.set_ylabel(r"Model stance shift  ($\bar{Y}_k-\bar{Y}_{baseline}$)", fontsize=11.5)

    # Model = marker shape; cue family = colour. Public build adds the logo to the key.
    if USE_LOGOS:
        leg1 = logo_shape_legend(ax, MODELS, MODEL_MARKER, loc="upper left", frameon=False,
                                 fontsize=9.5, handlelength=2.6, handletextpad=0.6,
                                 labelspacing=0.7)
    else:
        mh = [plt.Line2D([], [], marker=MODEL_MARKER[m], ls="", color="#444444",
                         mec="white", mew=0.5, label=MODEL_LABEL[m]) for m in MODELS]
        leg1 = ax.legend(handles=mh, loc="upper left", frameon=False, fontsize=9.5)
    ax.add_artist(leg1)
    fam_handles = [plt.Line2D([], [], marker="s", ls="", color=FAM_COLOUR[f], ms=9,
                              label=FAM_MARKER_LABEL[f], mec="white", mew=0.5)
                   for f in FAM_MARKER]
    ax.legend(handles=fam_handles, loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    _save(fig, out, "fig2_calibration", fmts)


# --------------------------------------------------------------------------- #
# Fig 3 — baseline Conservative / Neutral / Liberal composition, per issue
# --------------------------------------------------------------------------- #
def _composition(df, mask):
    s = df[mask][SCORE]
    n = len(s)
    if n == 0:
        return 0.0, 0.0, 0.0
    return (s == -1).sum() / n, (s == 0).sum() / n, (s == 1).sum() / n


def load_issue_labels(issues_csv: Path) -> dict[str, str]:
    """Map issue_id (= ces_variable) -> short human label for row headers."""
    iss = pd.read_csv(issues_csv)
    return dict(zip(iss["ces_variable"], iss["ces_item_short"]))


def load_liberal_sign(issues_csv: Path) -> dict[str, int]:
    """Map issue_id -> liberal_sign. +1 => supporting the issue is the liberal
    position (so the liberal side is the "support/pro" side); -1 => supporting it
    is the conservative position (conservative side is the "support/pro" side)."""
    iss = pd.read_csv(issues_csv)
    return dict(zip(iss["ces_variable"], iss["liberal_sign"].astype(int)))


def _pct_ints(fracs):
    """Round fractions to integer percents that sum to exactly 100
    (largest-remainder), so the three labels in a stack never read 99/101%."""
    raw = [f * 100 for f in fracs]
    out = [int(np.floor(x)) for x in raw]
    short = 100 - sum(out)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[:short]:
        out[i] += 1
    return out


def fig_composition(data, out: Path, fmts, issues_csv: Path):
    """One row per issue, one column per model, no-cue (baseline) condition only.

    The stacked Con/Neu/Lib mix at baseline shows each issue's starting stance mix
    per model, which is the ceiling/floor context for the shifts in Fig 1: an issue
    already saturated on one side has little room left to move."""
    labels = load_issue_labels(issues_csv)
    lib_sign = load_liberal_sign(issues_csv)
    # Keep only issues that were actually generated (the run is the 19 main
    # items; the CSV also lists sensitivity/robustness items that weren't run).
    present = set.union(*(set(data[m][base_mask(data[m])].issue_id.unique()) for m in MODELS))
    labels = {iss: lbl for iss, lbl in labels.items() if iss in present}
    # Baseline composition per (model, issue), computed once.
    comp = {m: {iss: _composition(data[m], base_mask(data[m]) & (data[m].issue_id == iss))
                for iss in labels}
            for m in MODELS}
    # Order issues by mean net-liberal lean (lib - con) across models, most
    # liberal at top -> most conservative at bottom, so ceilings cluster.
    issues = sorted(labels,
                    key=lambda iss: np.mean([comp[m][iss][2] - comp[m][iss][0] for m in MODELS]),
                    reverse=True)

    # Wrapping the few long issue names keeps the label gutter compact when the
    # vector figure is scaled to the thesis text width.  Break at words only;
    # short labels remain on one line.
    display_labels = {
        iss: textwrap.fill(label, width=24, break_long_words=False,
                           break_on_hyphens=False)
        for iss, label in labels.items()
    }

    # Keep the physical canvas close to the rendered \linewidth so the fonts
    # survive scaling: ~8.75in wide -> ~0.69 scale at a 6in text block.
    fig, axes = plt.subplots(len(issues), len(MODELS),
                             figsize=(1.75 * len(MODELS), 0.42 * len(issues) + 1.5),
                             squeeze=False)
    for j, m in enumerate(MODELS):
        for i, iss in enumerate(issues):
            ax = axes[i][j]
            con, neu, lib = comp[m][iss]
            # Liberal on the LEFT, conservative on the RIGHT.
            left = 0.0
            for frac, pct, colour, txtcol in zip(
                    (lib, neu, con), _pct_ints((lib, neu, con)),
                    (C_LIB, C_NEU, C_CON), ("white", "#444444", "white")):
                ax.barh(0, frac, left=left, height=0.70, color=colour)
                if frac >= 0.22:  # label only segments wide enough to hold it
                    ax.text(left + frac / 2, 0, f"{pct}%", ha="center", va="center",
                            fontsize=9, color=txtcol)
                left += frac
            ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=9.2, pad=8)
            if j == 0:
                # Thumbs-up marks the "support/pro" side of the issue, coloured by
                # which side that is (blue = liberal supports, red = conservative).
                thumb_col = C_LIB if lib_sign[iss] > 0 else C_CON
                ax.plot(-0.07, 0.0, marker=THUMB, markersize=9, color=thumb_col,
                        ls="", transform=ax.get_yaxis_transform(), clip_on=False)
                ax.text(-0.15, 0.0, display_labels[iss], rotation=0,
                        ha="right", va="center", multialignment="right",
                        fontsize=10, fontweight="bold", linespacing=0.92,
                        transform=ax.get_yaxis_transform(), clip_on=False)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=C_LIB, label="Liberal"),
               plt.Line2D([], [], marker="s", ls="", ms=11, color=C_NEU, label="Neutral"),
               plt.Line2D([], [], marker="s", ls="", ms=11, color=C_CON, label="Conservative")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.01))
    # Second row: what the coloured thumbs-up on each issue means.
    thumb_handles = [
        plt.Line2D([], [], marker=THUMB, ls="", ms=10, color=C_LIB,
                   label="liberal side supports the issue"),
        plt.Line2D([], [], marker=THUMB, ls="", ms=10, color=C_CON,
                   label="conservative side supports the issue")]
    fig.legend(handles=thumb_handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.045))
    fig.subplots_adjust(left=0.335, right=0.98, top=0.95, bottom=0.075,
                        hspace=0.45, wspace=0.14)
    _save(fig, out, "fig3_composition", fmts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full_3x")
    ap.add_argument("--ces-table", default="results/full_3x/ces_estimates.csv")
    ap.add_argument("--issues-csv", default="data/input/issues_experiment.csv")
    ap.add_argument("--robust-dir", default="results/robustness")
    ap.add_argument("--figures-dir", default="figures/full_3x")
    ap.add_argument("--format", default="both", choices=["pdf", "png", "both"],
                    help="output format(s); pdf is vector, best for LaTeX (default both)")
    args = ap.parse_args()
    fmts = ["pdf", "png"] if args.format == "both" else [args.format]
    rd, fd = Path(args.results_dir), Path(args.figures_dir)
    # Public/logo build writes to its own dir so the shape-based thesis figures stay put.
    if USE_LOGOS and args.figures_dir == "figures/full_3x":
        fd = Path("figures/full_3x_logos")
    fd.mkdir(parents=True, exist_ok=True)
    data = load(rd)
    fig_forest(data, fd, fmts)
    fig_forest_panels(data, fd, fmts)  # fig1 split into subfigures (a) and (b)
    fig_calibration(data, Path(args.ces_table), fd, fmts, Path(args.robust_dir))
    fig_composition(data, fd, fmts, Path(args.issues_csv))
    print(f"Wrote 5 figures to {fd} as {', '.join(fmts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
