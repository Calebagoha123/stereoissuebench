#!/usr/bin/env python3
"""RQ3 'ladder' figures (§4.4): the four-step chain as four standalone figures.

  belief_vs_stance : belief shift vs written-stance shift, one point per cue group,
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

PROBE = Path("results/probe_internal")
BELIEF = Path("results/full")
MASTER = Path("results/consolidated/01_master_cue_effects.csv")
OUTDIR = Path("figures/probe_thesis")

# Open-weight models carry both arms (behavioural + mechanistic). The two hosted
# frontier models expose no internals, so they appear in the *behavioural* figures
# only (belief-vs-stance, relevance, direct-refusal), not the mechanistic ones.
MODELS_ORDER = ["llama", "gemma", "qwen"]
MODELS_BEHAV = ["llama", "gemma", "qwen", "gpt56terra", "sonnet5"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B", "qwen": "Qwen3.6-27B",
               "gpt56terra": "GPT-5.6", "sonnet5": "Sonnet 5"}

# cue-family palette (consistent with the DiD / cross-model figures)
FAM_COLOUR = {
    "explicit_political": "#1F3A93",    # Oxford navy
    "explicit_demographic": "#C7372F",
    "implicit_political": "#58A9DE",
    "implicit_demographic": "#F0821E",
}
FAM_LABEL = {
    "explicit_political": "Explicit political",
    "explicit_demographic": "Explicit demographic",
    "implicit_political": "Implicit political (state)",
    "implicit_demographic": "Implicit demographic (name)",
}

INK = "#1a1a1a"
GRID = "#d9d9d9"

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
    ax.set_xlabel("belief shift (probe, cued − baseline)")
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
    colours = ["#C7372F" if a == "name" else "#5b6673" for a in m.index]
    ax.barh(range(len(m)), m.values, color=colours, height=0.68)
    for i, v in enumerate(m.values):
        ax.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("mean relevance for predicting opinion (0–100)")
    ax.spines["left"].set_visible(False)


def c_transfer(ax, model):
    tr = pd.read_csv(PROBE / f"{model}_cross_cue_transfer.csv")
    summ = json.loads((PROBE / f"{model}_summary.json").read_text())
    chance = float(summ.get("transfer_chance", 0.25))
    label_to_name = float(summ.get("transfer_label_to_name_max", tr["label_to_name"].max()))
    within = 1.0  # within-family decodability ceiling (summary decodability_best)
    bars = [("within-family\n(race×gender)", within, "#5b6673"),
            ("cross-cue transfer\n(label → name)", label_to_name, "#1F3A93")]
    x = range(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.axhline(chance, color="#C7372F", ls="--", lw=1.2)
    ax.text(len(bars) - 0.5, chance + 0.015, f"chance = {chance:.2f}", ha="right",
            fontsize=8, color="#C7372F")
    ax.set_xticks(list(x)); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("balanced accuracy")


def d_refusal(ax, model):
    d = _load_direct_labels(model)
    order = ["gender", "race", "political"]
    disp = {"gender": "gender", "race": "race", "political": "politics"}
    seg = ["committed", "committed_with_caveat", "refused"]
    seg_col = {"committed": "#2f7d4f", "committed_with_caveat": "#8cc79e",
               "other": "#b8b8b8", "refused": "#C7372F"}
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


def threeup_belief(models):
    """Belief vs stance, all behavioural models side by side (shared axes, one legend).

    Open-weight models plus the two hosted frontier models; the frontier panels are
    marked with a rule under the title since they carry the behavioural arm only."""
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
        ax.set_xlabel("belief shift (cued − baseline)")
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
    save_fig(fig, OUTDIR / "rq3_belief_vs_stance_3up")


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
        colours = ["#C7372F" if a == "name" else "#5b6673" for a in order]
        ax.barh(range(len(order)), vals, color=colours, height=0.66)
        for i, v in enumerate(vals):
            ax.text(v + 2, i, f"{v:.0f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
        ax.set_xlim(0, 100); ax.set_title(MODEL_LABEL[m], fontsize=12)
        ax.set_xlabel("mean relevance (0–100)")
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, OUTDIR / "rq3_relevance_3up")


def threeup_mediation(models):
    """Internal political-axis shift vs written-stance shift, 3 models side by side.

    B3 mediation: projection of each cue group onto the per-layer Democrat-Republican
    activation direction (proj_shift) predicts the written-stance shift.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import stats
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=False)
    for ax, m in zip(axes, models):
        d = pd.read_csv(PROBE / f"{m}_mediation_full3x.csv")
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
        ax.set_xlabel("internal political-axis shift\n(standardised, cued − baseline)")
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


def threeup_transfer(models):
    """Per-name-group cross-cue transfer (label → name) as a depth heatmap.

    Rows = race×gender name groups, columns = layer depth, colour = recall of a
    probe trained on the *explicit* demographic label and tested on the *name*
    cue alone (3-layer smoothed). The colour scale diverges around 4-class chance
    (0.25): red = the group's identity is linearly recoverable from its names at
    that depth, blue = not. A single "best layer" bar is unstable — the same
    4-class balanced accuracy is reached at layers with very different per-group
    confusions (e.g. qwen Black-man recall swings 0.02→0.94 by depth) — so the
    full depth profile is shown. It reveals that *which* name group is decodable,
    and *where*, is model-specific rather than a fixed name hierarchy.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    chance = 0.25
    pivs = {m: pd.read_csv(PROBE / f"{m}_transfer_by_group.csv")
            .pivot(index="cue_group", columns="layer", values="label_to_name_recall")
            .reindex(NAME_GROUP_ORDER) for m in models}
    widths = [pivs[m].shape[1] for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.0), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    norm = TwoSlopeNorm(vmin=0.0, vcenter=chance, vmax=1.0)
    im = None
    for ax, m in zip(axes, models):
        sm = pivs[m].T.rolling(3, center=True, min_periods=1).mean().T  # smooth over layers
        im = ax.imshow(sm.values, aspect="auto", cmap="RdBu_r", norm=norm,
                       extent=[0, 1, len(NAME_GROUP_ORDER) - 0.5, -0.5],
                       interpolation="nearest")
        ax.set_title(MODEL_LABEL[m], fontsize=12)
        ax.set_xlabel("relative depth (layer / final)")
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks(range(len(NAME_GROUP_ORDER)))
        ax.tick_params(length=0)
    axes[0].set_yticklabels([NAME_GROUP_LABEL[g] for g in NAME_GROUP_ORDER])
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.028, pad=0.02)
    cbar.set_label("identity recovered from name\n(label → name recall)", fontsize=9.5)
    cbar.set_ticks([0, chance, 0.5, 1.0])
    cbar.ax.axhline(chance, color=INK, lw=0.9)  # mark chance on the ramp
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
    seg_col = {"committed": "#2f7d4f", "committed_with_caveat": "#8cc79e",
               "other": "#b8b8b8", "refused": "#C7372F"}
    seg_lab = {"committed": "answered", "committed_with_caveat": "answered w/ caveat",
               "other": "unclear", "refused": "refused"}
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.6, 3.4), sharey=True)
    for ax, m in zip(axes, models):
        d = _load_direct_labels(m)
        ct = pd.crosstab(d.attribute, d.label).reindex(index=order, columns=seg, fill_value=0)
        prop = ct.div(ct.sum(axis=1), axis=0)[seg]
        prop.index = [disp[a] for a in prop.index]
        prop.iloc[::-1].plot.barh(stacked=True, ax=ax, width=0.62, legend=False,
                                  color=[seg_col[s] for s in seg])
        ax.set_xlim(0, 1); ax.set_title(MODEL_LABEL[m], fontsize=12)
        ax.set_xlabel("share of direct-probe responses"); ax.set_ylabel("")
        ax.spines["left"].set_visible(False)
    handles = [mpatches.Patch(color=seg_col[s]) for s in seg]
    fig.legend(handles, [seg_lab[s] for s in seg], loc="lower center", ncol=4,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.threeup:
        # behavioural arm: all five models (open-weight + frontier)
        threeup_belief(MODELS_BEHAV)
        threeup_relevance(MODELS_BEHAV)
        threeup_refusal(MODELS_BEHAV)
        # mechanistic arm: open-weight models only (frontier expose no internals)
        threeup_mediation(MODELS_ORDER)
        threeup_transfer(MODELS_ORDER)
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
