#!/usr/bin/env python3
"""RQ3 'ladder' figures (§4.4): the four-step chain as four standalone figures.

  belief_vs_stance : predicted-opinion shift vs written-stance shift, one point per cue group,
      y=x reference -> models write toward belief but under-write it.
  relevance        : mean relevance rating per attribute (0-100) -> a name is
      rated near-useless for predicting opinion.
  transfer         : internal decodability, within-family (ceiling) vs cross-cue
      transfer (label->name) against chance -> names are encoded even so.
  direct_refusal   : direct-probe commit/refuse by attribute -> the name->politics
      inference is the one refused when asked outright.

Emits four separate figures by default (each gets its own thesis caption); pass
--composite to also write the single 2x2 panel. Llama is the worked example
(--model llama); --model {gemma,qwen} gives the Appendix-C cross-model versions.
Reads the probe outputs already in results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _markers import marker_ms, scatter_s  # equal-ink marker sizing (see _markers.py)
import _style  # for TIMES: cmr10 has no unicode multiplication sign

PROBE = Path("results/probe_internal")
BELIEF = Path("results/full")
MASTER = Path("results/consolidated/01_master_cue_effects.csv")
OUTDIR = Path("figures/probe_thesis")

# Open-weight models carry both arms (behavioural + mechanistic). The two hosted
# frontier models expose no internals, so they appear in the *behavioural* figures
# only (belief-vs-stance, relevance, direct-refusal), not the mechanistic ones.
MODELS_ORDER = ["llama", "gemma", "qwen"]
MODELS_BEHAV = ["llama", "gemma", "qwen", "gpt56terra", "sonnet5"]
MODEL_LABEL = {"qwen": "Qwen-3.6-27B", "gemma": "Gemma-3-12B", "llama": "Llama-3.1-8B",
               "gpt56terra": "GPT-5.6 Terra", "sonnet5": "Claude Sonnet 5"}
# Document-wide model vocabulary, identical to make_thesis_figures.py /
# make_ces_dumbbell.py: Okabe-Ito colour + a redundant marker shape so model identity
# survives greyscale printing. Do not diverge from it here -- a reader who has learnt
# "blue triangle = Llama" from Figure 1 must not have to relearn it in §4.4.
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#0072B2",
                "gpt56terra": "#CC79A7", "sonnet5": "#56B4E9"}
MODEL_MARKER = {"qwen": "o", "gemma": "s", "llama": "^", "gpt56terra": "D", "sonnet5": "v"}

# cue-family palette: Okabe-Ito, chosen distinct from the model colours (below) so
# fill=family / edge=model encodings never collide.
FAM_COLOUR = {
    "explicit_political": "#D55E00",    # vermillion
    "explicit_demographic": "#CC79A7",  # reddish purple
    "implicit_political": "#56B4E9",    # sky blue
    "implicit_demographic": "#000000",  # black (the focal name cue / null)
}
# Cue-type names, verbatim from the FOREST_BANDS vocabulary in make_thesis_figures.py
# (Figure 1). The explicit/implicit 2x2 was dropped in favour of naming the four cue
# types by what they are and ordering them by directness, so these figures must not
# reintroduce the old wording. Data keys are unchanged.
FAM_LABEL = {
    "explicit_political": "Party label",
    "explicit_demographic": "Race " + _style.TIMES + " gender",
    "implicit_political": "State",
    "implicit_demographic": "Name",
}

INK = "#1a1a1a"
GRID = "#d9d9d9"


def _darken(hex_colour: str, factor: float):
    """Blend a hex colour toward black (factor 1.0 = unchanged), for using the marker
    palette as small text without losing which hue is which. Mirrors the helper in
    make_thesis_figures.py -- Qwen's orange and Sonnet's sky blue are both too light
    to read as label text undarkened."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_colour)
    return (r * factor, g * factor, b * factor)

SAVE_FMTS = ["png"]  # set from --fmt in main()


def _load_direct_labels(model):
    """Prefer the adjudicated labels (relabel_direct_probe.py); fall back to the rule.

    The adjudicated sheet only covers the open-weight models; the hosted frontier
    models fall back to their pipeline `label` column in results/full/."""
    p = PROBE / "direct_probe_labeled.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.model == model].copy()
        if len(d):
            return d
    return pd.read_csv(BELIEF / f"direct_probe_{model}.csv", low_memory=False)


def belief_vs_stance(model):
    """One row per cue group: elicited belief shift vs written-stance shift.

    Belief shift is the continuous opinion-prediction shift (belief_probe, cued −
    baseline, oriented to the liberal side). The written-stance shift is the model's
    cue effect from the classifier of record (luna), read from the consolidated
    master table so all five models — including the two hosted frontier models that
    expose no internals — use the same stance ruler. Reproduces the r's reported in
    §4.4 (0.81 Llama, 0.78 Gemma, 0.70 Qwen, 0.77 GPT-5.6, 0.87 Sonnet 5)."""
    d = pd.read_csv(BELIEF / f"belief_probe_{model}.csv", low_memory=False)
    d = d[d.probe_kind == "opinion"].copy()
    d["score"] = pd.to_numeric(d["parsed_score"], errors="coerce")
    d = d.dropna(subset=["score"])
    d["b_cont"] = (d["score"] - 50) / 50 * d["liberal_sign"]
    base = d[d.cue_family == "baseline"].groupby("issue_id")["b_cont"].mean()
    rows = []
    for (fam, grp), c in d[d.cue_family != "baseline"].groupby(["cue_family", "cue_group"]):
        rows.append((fam, grp, (c.groupby("issue_id")["b_cont"].mean() - base).mean()))
    belief = pd.DataFrame(rows, columns=["cue_family", "cue_group", "belief_cont"])
    master = pd.read_csv(MASTER)
    master = master[master.model == model][["cue_family", "cue_group", "model_shift"]]
    out = belief.merge(master, on=["cue_family", "cue_group"], how="inner")
    return out.rename(columns={"model_shift": "stance_shift"})


def save_fig(fig, stem):
    """Save a figure to every requested format (stem has no extension)."""
    import matplotlib.pyplot as plt
    for ext in SAVE_FMTS:
        p = f"{stem}.{ext}"
        fig.savefig(p, dpi=230, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


def panel_label(ax, letter):
    ax.text(-0.14, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")


def a_belief_stance(ax, model):
    d = pd.read_csv(PROBE / "belief_under_acting.csv")
    d = d[d.model == model]
    from scipy import stats
    r = stats.pearsonr(d["belief_cont"], d["stance_shift"])[0]
    lim = 0.9
    ax.plot([-lim, lim], [-lim, lim], "--", color="#7a7a7a", lw=1.1, zorder=1)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    for fam, g in d.groupby("cue_family"):
        ax.scatter(g["belief_cont"], g["stance_shift"], s=46, color=FAM_COLOUR[fam],
                   edgecolor="white", linewidth=0.6, zorder=3, label=FAM_LABEL[fam])
    # annotate the Republican worked example
    rep = d[(d.cue_family == "explicit_political") & (d.cue_group == "republican")].iloc[0]
    ax.annotate(f"Republican:\nbelieves {rep['belief_cont']:+.2f},\nwrites {rep['stance_shift']:+.2f}",
                (rep["belief_cont"], rep["stance_shift"]),
                xytext=(rep["belief_cont"] + 0.06, rep["stance_shift"] - 0.28),
                fontsize=7.6, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.7))
    ax.text(0.44, 0.80, "$y=x$\n(writes = believes)", fontsize=7.8, color="#7a7a7a",
            rotation=45, ha="center", va="center", rotation_mode="anchor")
    ax.text(0.03, 0.95, f"$r = {r:.2f}$", transform=ax.transAxes, fontsize=10,
            va="top", ha="left")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(PRED_LABEL)
    ax.set_ylabel("written-stance shift")
    ax.legend(fontsize=8, frameon=False, loc="lower right", handletextpad=0.3,
              borderpad=0.2, labelspacing=0.3)


def b_relevance(ax, model):
    d = pd.read_csv(BELIEF / f"belief_probe_{model}.csv", low_memory=False)
    r = d[d.probe_kind == "relevance"].copy()
    r["s"] = pd.to_numeric(r["parsed_score"], errors="coerce")
    m = r.groupby("attribute")["s"].mean().sort_values()
    disp = {"party": "party", "state": "state", "gender": "gender", "race": "race", "name": "name"}
    labels = [disp.get(a, a) for a in m.index]
    colours = ["#D55E00" if a == "name" else "#7A7A7A" for a in m.index]
    ax.barh(range(len(m)), m.values, color=colours, height=0.68)
    for i, v in enumerate(m.values):
        ax.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("mean relevance for predicting opinion (0-100)")
    ax.spines["left"].set_visible(False)


def c_transfer(ax, model):
    tr = pd.read_csv(PROBE / f"{model}_cross_cue_transfer.csv")
    summ = json.loads((PROBE / f"{model}_summary.json").read_text())
    chance = float(summ.get("transfer_chance", 0.25))
    label_to_name = float(summ.get("transfer_label_to_name_max", tr["label_to_name"].max()))
    within = 1.0  # within-family decodability ceiling (summary decodability_best)
    bars = [("within-family\n(race $\\times$ gender)", within, "#7A7A7A"),
            ("cross-cue transfer\n(label $\\rightarrow$ name)", label_to_name, "#0072B2")]
    x = range(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.axhline(chance, color="#D55E00", ls="--", lw=1.2)
    ax.text(len(bars) - 0.5, chance + 0.015, f"chance = {chance:.2f}", ha="right",
            fontsize=8, color="#D55E00")
    ax.set_xticks(list(x)); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("balanced accuracy")


def d_refusal(ax, model):
    d = _load_direct_labels(model)
    order = ["gender", "race", "political"]
    disp = {"gender": "gender", "race": "race", "political": "politics"}
    seg = ["committed", "committed_with_caveat", "refused"]
    seg_col = {"committed": "#009E73", "committed_with_caveat": "#7FCBB4",
               "other": "#BBBBBB", "refused": "#D55E00"}
    seg_lab = {"committed": "answered", "committed_with_caveat": "answered w/ caveat",
               "other": "unclear", "refused": "refused"}
    ct = pd.crosstab(d.attribute, d.label).reindex(index=order, columns=seg, fill_value=0)
    prop = ct.div(ct.sum(axis=1), axis=0)[seg]
    prop.index = [disp[a] for a in prop.index]
    import matplotlib.patches as mpatches
    prop.iloc[::-1].plot.barh(stacked=True, ax=ax, width=0.62, legend=False,
                              color=[seg_col[s] for s in seg])
    handles = [mpatches.Patch(color=seg_col[s]) for s in seg]
    ax.legend(handles, [seg_lab[s] for s in seg], fontsize=8.5, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.25), handletextpad=0.4,
              columnspacing=1.2)
    ax.set_xlim(0, 1); ax.set_xlabel("share of direct-probe responses")
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)


#: axis wording for the elicited opinion-prediction quantity. Deliberately *not*
#: "belief": the probe records a number the model emits when asked to predict a
#: group's opinion, and calling that a belief imports a mental state the measurement
#: cannot license. "Predicted opinion" pairs with "written stance" on the other axis
#: -- both are model outputs, one about the user and one for the user.
PRED_LABEL = r"predicted-opinion shift (cued $-$ baseline)"
STANCE_LABEL = r"written-stance shift (cued $-$ baseline)"


def facet_belief_by_cue(models):
    """Predicted opinion vs written stance, one panel per *cue type*, colour = model.

    Faceting choice. The earlier version put models in panels and cue types in colour,
    which repeated the same diagonal shape five times and buried the finding: the state
    cue's dissociation was three dots inside a cloud. Panelling by cue type instead
    makes each panel a claim about one cue -- the party label spans the diagonal, state
    collapses to a flat band at y = 0 with x spread over [-0.5, +0.5], name collapses to
    the origin -- and turns the model comparison into a cheap within-panel one. The cost
    is that per-model r, fitted over all 14 groups at once, belongs to no panel: it is
    reported in the text rather than on the figure.

    Panel order follows FOREST_BANDS (Figure 1): party label, race x gender, state,
    name, i.e. by decreasing directness.

    2x2 rather than 1x4 so each panel gets roughly twice the linear size at the same
    \\linewidth, keeping the figure upright (no sidewaysfigure needed)."""
    import matplotlib.pyplot as plt
    fams = ["explicit_political", "explicit_demographic",
            "implicit_political", "implicit_demographic"]
    data = {m: belief_vs_stance(m) for m in models}
    lim = 0.9
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 9.6), sharex=True, sharey=True)
    for ax, fam in zip(axes.ravel(), fams):
        # The attenuation region is the wedge between y = x and y = 0, in *both*
        # signed quadrants: under-writing means a shallower slope than the diagonal,
        # not "below the diagonal" (a Republican cue is written less negatively, so it
        # sits *above* it). Shading both wedges states the asymmetry correctly.
        ax.fill([0, lim, lim], [0, 0, lim], color="#f0f0f0", zorder=0, lw=0)
        ax.fill([0, -lim, -lim], [0, 0, -lim], color="#f0f0f0", zorder=0, lw=0)
        ax.plot([-lim, lim], [-lim, lim], "--", color="#9a9a9a", lw=1.0, zorder=1)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        for m in models:
            g = data[m][data[m].cue_family == fam]
            ax.scatter(g["belief_cont"], g["stance_shift"], marker=MODEL_MARKER[m],
                       s=scatter_s(MODEL_MARKER[m], 8.6), color=MODEL_COLOUR[m],
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_title(FAM_LABEL[fam], fontsize=12.5, color=INK, pad=7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    # y=x sits on the panel diagonal, which in axes-fraction coordinates runs (0,0) ->
    # (1,1): the label therefore goes at (a, a + eps), a small perpendicular nudge off
    # the line, not at an arbitrary height above it.
    _a = 0.74
    axes[0, 0].text(_a, _a + 0.022, "$y=x$", transform=axes[0, 0].transAxes,
                    fontsize=9.5, color="#8a8a8a", rotation=45, rotation_mode="anchor",
                    ha="center", va="bottom")
    for ax in axes[1, :]:
        ax.set_xlabel(PRED_LABEL, fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(STANCE_LABEL, fontsize=11)
    # No r on the figure. Per model it is a 14-group quantity that belongs to no panel;
    # per cue type within a model it is fitted on 3-4 points and pinned near 1 even for
    # the name cue (0.72-0.98), which would advertise a strong relationship exactly
    # where the finding is that nothing transmits. The five per-model values stay in the
    # text; what the panels carry is the transmission slope, which is the on-claim
    # statistic (see the caption).
    handles = [plt.Line2D([], [], marker=MODEL_MARKER[m], ls="",
                          ms=marker_ms(MODEL_MARKER[m], 7.0), color=MODEL_COLOUR[m],
                          mec="white", mew=0.7, label=MODEL_LABEL[m])
               for m in models]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.004),
               handletextpad=0.5, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=2.6)
    save_fig(fig, OUTDIR / "rq3_belief_vs_stance_3up")


def threeup_belief(models):
    """Belief vs stance, all behavioural models side by side (shared axes, one legend).

    Superseded by facet_belief_by_cue (panels = cue type); kept behind
    ``--facet models`` for comparison."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import stats
    lim = 0.9
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.6, 4.4), sharex=True, sharey=True)
    for ax, m in zip(axes, models):
        d = belief_vs_stance(m)
        r = stats.pearsonr(d["belief_cont"], d["stance_shift"])[0]
        ax.plot([-lim, lim], [-lim, lim], "--", color="#7a7a7a", lw=1.1, zorder=1)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        for fam, g in d.groupby("cue_family"):
            ax.scatter(g["belief_cont"], g["stance_shift"], s=42, color=FAM_COLOUR[fam],
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(0.05, 0.95, f"$r = {r:.2f}$", transform=ax.transAxes, fontsize=11, va="top")
        ax.set_title(MODEL_LABEL[m], fontsize=12)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel(PRED_LABEL, fontsize=10)
    axes[0].set_ylabel("written-stance shift")
    axes[-1].text(0.62, 0.72, "$y=x$", transform=axes[-1].transAxes, fontsize=9,
                  color="#7a7a7a", rotation=45, rotation_mode="anchor")
    handles = [mpatches.Patch(color=FAM_COLOUR[f]) for f in
               ["explicit_political", "explicit_demographic", "implicit_demographic", "implicit_political"]]
    labels = [FAM_LABEL[f] for f in
              ["explicit_political", "explicit_demographic", "implicit_demographic", "implicit_political"]]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, OUTDIR / "rq3_belief_vs_stance_bymodel")


def threeup_relevance(models):
    """Relevance bars, three open models side by side, fixed attribute order."""
    import matplotlib.pyplot as plt
    order = ["name", "race", "gender", "state", "party"]  # bottom -> top
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.6, 3.7), sharey=True)
    for ax, m in zip(axes, models):
        d = pd.read_csv(BELIEF / f"belief_probe_{m}.csv", low_memory=False)
        r = d[d.probe_kind == "relevance"].copy()
        r["s"] = pd.to_numeric(r["parsed_score"], errors="coerce")
        mean = r.groupby("attribute")["s"].mean()
        vals = [mean.get(a, float("nan")) for a in order]
        colours = ["#D55E00" if a == "name" else "#7A7A7A" for a in order]
        ax.barh(range(len(order)), vals, color=colours, height=0.66)
        for i, v in enumerate(vals):
            ax.text(v + 2, i, f"{v:.0f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
        ax.set_xlim(0, 100); ax.set_title(MODEL_LABEL[m], fontsize=12)
        ax.set_xlabel("mean relevance (0-100)")
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, OUTDIR / "rq3_relevance_3up")


def threeup_mediation(models):
    """Internal political-axis shift vs written-stance shift, 3 models side by side.

    B3 mediation: projection of each cue group onto the per-layer Democrat-Republican
    activation direction (proj_shift) predicts the written-stance shift. The stance
    side is the classifier-of-record (luna) shift from the consolidated master, not
    mediation_full3x's stale `stance_shift` column, so the panel r's match §4.4.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import stats
    master = pd.read_csv(MASTER)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=False)
    for ax, m in zip(axes, models):
        d = pd.read_csv(PROBE / f"{m}_mediation_full3x.csv").drop(columns=["stance_shift"])
        mm = master[master.model == m][["cue_family", "cue_group", "model_shift"]]
        d = d.merge(mm, on=["cue_family", "cue_group"], how="inner").rename(
            columns={"model_shift": "stance_shift"})
        raw, y = d["proj_shift"].to_numpy(), d["stance_shift"].to_numpy()
        # projection units are model-specific (activation scale); standardise x so the
        # three panels are comparable. Correlation and fit are unchanged by z-scoring.
        x = (raw - raw.mean()) / raw.std(ddof=0)
        d = d.assign(_z=x)
        r = stats.pearsonr(x, y)[0]
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b * xs + a, color="#555", lw=1.1, zorder=1)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        for fam, g in d.groupby("cue_family"):
            ax.scatter(g["_z"], g["stance_shift"], s=44, color=FAM_COLOUR[fam],
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(0.04, 0.95, f"$r = {r:.2f}$", transform=ax.transAxes, fontsize=11, va="top")
        ax.set_title(MODEL_LABEL[m], fontsize=12)
        ax.set_xlim(-2.6, 2.6)
        ax.set_xlabel(r"internal political-axis shift" "\n" r"(standardised, cued $-$ baseline)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("written-stance shift")
    handles = [mpatches.Patch(color=FAM_COLOUR[f]) for f in
               ["explicit_political", "explicit_demographic", "implicit_demographic", "implicit_political"]]
    labels = [FAM_LABEL[f] for f in
              ["explicit_political", "explicit_demographic", "implicit_demographic", "implicit_political"]]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, OUTDIR / "rq3_mediation_3up")


NAME_GROUP_LABEL = {"black_man": "Black man", "black_woman": "Black woman",
                    "white_man": "White man", "white_woman": "White woman"}
NAME_GROUP_ORDER = ["black_man", "black_woman", "white_man", "white_woman"]


# rows ordered by pooled decodability (women top, men bottom) so the gendered
# gradient reads top-to-bottom; see threeup_transfer docstring.
TRANSFER_ROW_ORDER = ["white_woman", "black_woman", "white_man", "black_man"]


def threeup_transfer(models):
    """Per-name-group cross-cue transfer (label → name) as a group breakdown.

    A probe trained on the *explicit* demographic label and tested on the *name* cue
    alone recovers the name's race×gender identity. We report, per name group and model,
    the share of network layers at which that recall beats 4-class chance (0.25) — a
    layer-agnostic "how consistently decodable" score (embedding layer 0 excluded, its
    recall being a raw-token artifact that inflates e.g. llama Black-man). We drop the
    per-layer depth profile deliberately: the *depth* at which a group becomes decodable
    is unstable and not of interest here — only *that* names encode identity, and *which*
    groups, is. The breakdown shows a gender-first, race-second ordering: white-woman
    names are the most linearly recoverable in every model, black-man names the least
    (pooled: women 0.65 > men 0.40; white 0.61 > Black 0.44); llama's Black-man bar is
    the lone exception.
    """
    import matplotlib.pyplot as plt
    chance = 0.25
    order = TRANSFER_ROW_ORDER  # women top, men bottom
    pivs = {m: pd.read_csv(PROBE / f"{m}_transfer_by_group.csv")
            .pivot(index="cue_group", columns="layer", values="label_to_name_recall")
            .reindex(order) for m in models}
    # per-group decodability: share of layers above chance, excluding the embedding layer
    summ = {m: {g: np.mean(pivs[m].loc[g].values[1:] > chance) for g in order}
            for m in models}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    h = 0.8 / len(models)
    for j, m in enumerate(models):
        ys = [i + ((len(models) - 1) / 2 - j) * h for i in range(len(order))]
        ax.barh(ys, [summ[m][g] for g in order], height=h, color=MODEL_COLOUR[m],
                label=MODEL_LABEL[m], edgecolor="white", linewidth=0.5)
        for yi, g in zip(ys, order):
            ax.text(summ[m][g] + 0.012, yi, f"{summ[m][g]:.2f}", va="center",
                    fontsize=9, color="#555")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME_GROUP_LABEL[g] for g in order], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"name $\rightarrow$ identity decodable" "\n" r"(share of layers above chance)",
                  fontsize=11)
    # marker + colour in the legend so this figure's model key matches every other
    # one in the thesis, even though the marks themselves are bars
    handles = [plt.Line2D([], [], marker=MODEL_MARKER[m], ls="",
                          ms=marker_ms(MODEL_MARKER[m], 6.6), color=MODEL_COLOUR[m],
                          mec="white", mew=0.6, label=MODEL_LABEL[m]) for m in models]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc="lower right",
              handlelength=1.1, handletextpad=0.5, labelspacing=0.35)
    fig.tight_layout()
    save_fig(fig, OUTDIR / "rq3_transfer_3up")


def threeup_refusal(models):
    """Direct-probe commit/refuse by attribute, three open models side by side."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    order = ["gender", "race", "political"]
    disp = {"gender": "gender", "race": "race", "political": "politics"}
    # keep "other" in the segment set so the refused share is normalised over *all*
    # responses (matches the rates quoted in §4.4); it is a thin grey sliver at most.
    seg = ["committed", "committed_with_caveat", "other", "refused"]
    # A neutral grey ramp for the answered grades plus Okabe-Ito vermillion for the
    # refusal. The earlier green/light-green pair collided with the document's model
    # vocabulary (#009E73 *is* Gemma), which reads as a model encoding in a figure
    # whose panels are already models. Greys carry the ordered "how fully answered"
    # scale, and the one saturated colour is reserved for the quantity of interest.
    seg_col = {"committed": "#3d3d3d", "committed_with_caveat": "#9c9c9c",
               "other": "#dedede", "refused": "#D55E00"}
    seg_lab = {"committed": "answered", "committed_with_caveat": "answered w/ caveat",
               "other": "unclear", "refused": "refused"}
    # Panel by *attribute*, rows by model -- same reasoning as facet_belief_by_cue: the
    # claim is "of the three inferences, the political one is the one refused", so the
    # attribute earns the panel and the model becomes a cheap within-panel comparison.
    # Also 3 panels rather than 5, so the figure is no longer a letterbox at \linewidth.
    props = {}
    for m in models:
        d = _load_direct_labels(m)
        ct = pd.crosstab(d.attribute, d.label).reindex(index=order, columns=seg, fill_value=0)
        props[m] = ct.div(ct.sum(axis=1), axis=0)[seg]
    rows = list(models)[::-1]           # first model at the top of each panel
    fig, axes = plt.subplots(1, len(order), figsize=(11.4, 4.6), sharey=True)
    for ax, attr in zip(axes, order):
        left = np.zeros(len(rows))
        ys = np.arange(len(rows))
        for s in seg:
            vals = np.array([props[m].loc[attr, s] for m in rows])
            ax.barh(ys, vals, left=left, height=0.62, color=seg_col[s],
                    edgecolor="white", linewidth=0.5, zorder=2)
            left += vals
        # the refused share is the quantity of interest: print it on the bar
        for y, m in zip(ys, rows):
            r = props[m].loc[attr, "refused"]
            ax.text(1.012, y, f"{r:.2f}", va="center", fontsize=9,
                    color="#8a4a10" if r > 0.5 else "#777")
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(disp[attr], fontsize=13, color=INK, pad=8)
        ax.set_xlabel("share of direct-probe responses", fontsize=10.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(rows)))
    axes[0].set_yticklabels([MODEL_LABEL[m] for m in rows], fontsize=11)
    for lab, m in zip(axes[0].get_yticklabels(), rows):
        lab.set_color(_darken(MODEL_COLOUR[m], 0.70))   # match the document model key
    handles = [mpatches.Patch(color=seg_col[s]) for s in seg]
    fig.legend(handles, [seg_lab[s] for s in seg], loc="lower center", ncol=4,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.2)
    save_fig(fig, OUTDIR / "rq3_direct_refusal_3up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=["llama", "gemma", "qwen"])
    ap.add_argument("--composite", action="store_true",
                    help="also emit the single 2x2 ladder figure")
    ap.add_argument("--threeup", action="store_true",
                    help="emit the two 3-model side-by-side figures (belief, relevance)")
    ap.add_argument("--fmt", default="both", choices=["png", "pdf", "both"],
                    help="output format(s); pdf is vector for LaTeX")
    ap.add_argument("--facet", default="cue", choices=["cue", "models", "both"],
                    help="belief-vs-stance panelling: by cue type (default, shipped) "
                         "or the old by-model version; 'both' emits each")
    args = ap.parse_args()
    m = args.model
    global SAVE_FMTS
    SAVE_FMTS = ["png", "pdf"] if args.fmt == "both" else [args.fmt]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#666", "axes.linewidth": 0.8,
        "xtick.color": "#444", "ytick.color": "#444",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType for LaTeX
    })
    import sys as _s2, pathlib as _p2
    _s2.path.insert(0, str(_p2.Path(__file__).resolve().parent))
    import _style
    _style.apply(plt)  # Computer Modern, to match the thesis document
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.threeup:
        # behavioural arm: all five models (open-weight + frontier)
        if args.facet in ("cue", "both"):
            facet_belief_by_cue(MODELS_BEHAV)
        if args.facet in ("models", "both"):
            threeup_belief(MODELS_BEHAV)
        threeup_refusal(MODELS_BEHAV)
        # mechanistic arm: open-weight models only (frontier expose no internals)
        threeup_transfer(MODELS_ORDER)
        # Demoted to the appendix (relevance: rank-identical to the prediction shift in
        # all five models; mediation: cue-presence offset + Republican leverage, see
        # ladder_summary.md). Still built, no longer shipped in §4.4.
        threeup_relevance(MODELS_BEHAV)
        threeup_mediation(MODELS_ORDER)
        return

    # four standalone figures, each with its own thesis caption
    panels = [
        (a_belief_stance, (5.4, 5.4), f"rq3_belief_vs_stance_{m}"),
        (b_relevance,     (6.2, 3.6), f"rq3_relevance_{m}"),
        (c_transfer,      (5.0, 4.4), f"rq3_transfer_{m}"),
        (d_refusal,       (6.4, 3.6), f"rq3_direct_refusal_{m}"),
    ]
    for fn, size, stem in panels:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, m)
        fig.tight_layout()
        save_fig(fig, OUTDIR / stem)

    if args.composite:
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4))
        a_belief_stance(axes[0, 0], m); panel_label(axes[0, 0], "a")
        b_relevance(axes[0, 1], m); panel_label(axes[0, 1], "b")
        c_transfer(axes[1, 0], m); panel_label(axes[1, 0], "c")
        d_refusal(axes[1, 1], m); panel_label(axes[1, 1], "d")
        fig.tight_layout(w_pad=3.0, h_pad=4.0)
        save_fig(fig, OUTDIR / f"rq3_ladder_{m}")


if __name__ == "__main__":
    main()
