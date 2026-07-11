#!/usr/bin/env python3
"""Thesis figures + table for the probe arm (the explanatory ladder).

Same house style as make_thesis_figures.py: no titles (a LaTeX caption carries
the takeaway), Okabe-Ito model colours, PDF (vector, for LaTeX) + PNG output.

The probe decomposes the implicit-cue null into four rungs — legibility (B1),
belief (A2), relevance (A3), use (B2/B3) — and asks where a name stops mattering.

  ladder_summary.{csv,md}     One row per model tracking the name cue down the
                              ladder: decodable and represented like an explicit
                              label (rungs 1-2), then collapsing at relevance/use.
  fig_p1_legibility_use.*     CENTREPIECE. Internal legibility (probe selectivity)
                              vs behavioural use (|written-stance shift|), per cue
                              family x model. Names are the most legible yet least
                              used — legibility does not buy use.
  fig_p2_transfer.*           Cross-cue probe transfer (train label -> test name,
                              and reverse) per model vs 4-way chance: the name
                              shares the explicit label's demographic direction.
  fig_p3_refusal.*            Direct-probe commit / caveat / refuse rates by
                              attribute x model: refusal tracks the sensitivity of
                              the inference (politics), not the clarity of the cue.

The internal political-axis projection (B2) and its B3 mediation are deliberately
NOT shipped as figures: proj_shift vs the no-cue baseline is confounded by a
cue-presence offset, and the mediation r is a leverage correlation carried by the
Republican outlier (see the ladder-table note). Internal-probe inputs
(results/probe_internal/<tag>_*) are unchanged by the 2k/3-rep regen (same prompts
-> same activations); only the stance side is full_3x.
"""
from __future__ import annotations

import argparse
import json
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

MODELS = ["qwen", "gemma", "llama"]
MODEL_LABEL = {"qwen": "Qwen-3.6-27B", "gemma": "Gemma-3-12B", "llama": "Llama-3.1-8B"}
MODEL_COLOUR = {"qwen": "#E69F00", "gemma": "#009E73", "llama": "#CC79A7"}  # Okabe-Ito

# cue family -> (decodability subset name, colour, short label)
FAM_SUBSET = {
    "explicit_political": "explicit_political",
    "explicit_demographic": "explicit_demographic",
    "implicit_political": "state",
    "implicit_demographic": "name",
}
FAM_COLOUR = {
    "explicit_political": "#1F3A93", "explicit_demographic": "#C7372F",
    "implicit_political": "#58A9DE", "implicit_demographic": "#F0821E",
}
FAM_LABEL = {
    "explicit_political": "Explicit political", "explicit_demographic": "Explicit demographic",
    "implicit_political": "Implicit political (state)", "implicit_demographic": "Implicit demographic (name)",
}
NAME_GROUPS = ["black_woman", "black_man", "white_woman", "white_man"]


def _save(fig, out: Path, stem: str, fmts) -> None:
    for ext in fmts:
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def summary(tag, pdir: Path) -> dict:
    return json.loads((pdir / f"{tag}_summary.json").read_text())


def stance_shifts(tag, sdir: Path) -> pd.DataFrame:
    """Per cue group: written-stance shift vs baseline, from full_3x DeBERTa."""
    df = pd.read_csv(sdir / f"bert_eval_{tag}.csv",
                     usecols=["arm", "cue_condition", "cue_family", "cue_group", "bert_liberal_score"],
                     low_memory=False)
    df["bert_liberal_score"] = pd.to_numeric(df["bert_liberal_score"], errors="coerce")
    base = df[(df.arm == "A") & (df.cue_condition == "baseline")]["bert_liberal_score"].mean()

    def cue_mask(fam, grp):
        if fam.startswith("explicit"):
            return (df.arm == "A") & (df.cue_condition == f"{fam}_{grp}")
        return (df.arm == "B") & (df.cue_family == fam) & (df.cue_group == grp)

    rows = []
    for fam in FAM_SUBSET:
        groups = df[df.cue_family == fam]["cue_group"].unique()
        for grp in groups:
            rows.append({"cue_family": fam, "cue_group": grp,
                         "stance_shift": df[cue_mask(fam, grp)]["bert_liberal_score"].mean() - base})
    return pd.DataFrame(rows)


# NOTE: the internal political-axis projection (B2) is intentionally NOT shipped
# as a figure. proj_shift vs the no-cue baseline is confounded by a large
# cue-presence offset (every cued group shifts the same way relative to a
# memory-free baseline), and the B3 mediation r is a leverage correlation carried
# almost entirely by the Republican outlier (r drops from ~0.8-0.9 to ~0.2-0.3
# once it is removed). It is reported with that caveat in the ladder table note
# and belongs in the text as a limitation, not a headline figure.


def belief_shifts(tag, bdir: Path) -> pd.DataFrame:
    b = pd.read_csv(bdir / f"belief_probe_{tag}.csv", low_memory=False)
    op = b[b["probe_kind"].eq("opinion")].copy()
    op["score"] = pd.to_numeric(op["parsed_score"], errors="coerce")
    op["sign"] = pd.to_numeric(op["liberal_sign"], errors="coerce")
    op["pred_lean"] = ((op["score"] - 50) / 50) * op["sign"]
    base = op.loc[op["cue_family"].eq("baseline"), "pred_lean"].mean()
    g = op.groupby(["cue_family", "cue_group"])["pred_lean"].mean().rename("belief_lean").reset_index()
    g["belief_shift"] = g["belief_lean"] - base
    return g


def relevance(tag, bdir: Path) -> dict:
    b = pd.read_csv(bdir / f"belief_probe_{tag}.csv", low_memory=False)
    rel = b[b["probe_kind"].eq("relevance")].copy()
    rel["score"] = pd.to_numeric(rel["parsed_score"], errors="coerce")
    return rel.groupby("attribute")["score"].mean().to_dict()


def direct_rates(tag, bdir: Path) -> pd.DataFrame:
    d = pd.read_csv(bdir / f"direct_probe_{tag}.csv", low_memory=False)
    out = (d.groupby(["attribute", "label"]).size() / d.groupby("attribute").size())
    return out.rename("rate").reset_index()


# --------------------------------------------------------------------------- #
# Table — the ladder
# --------------------------------------------------------------------------- #
def ladder_table(pdir: Path, sdir: Path, bdir: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for tag in MODELS:
        s = summary(tag, pdir)
        dec = s["decodability_best"]["name"]
        # full_3x mediation r, with and without the Republican leverage point
        med = pd.read_csv(pdir / f"{tag}_mediation_full3x.csv")
        med_r = float(np.corrcoef(med["proj_shift"], med["stance_shift"])[0, 1])
        med_norep = med[~((med.cue_family == "explicit_political") & (med.cue_group == "republican"))]
        med_r_norep = float(np.corrcoef(med_norep["proj_shift"], med_norep["stance_shift"])[0, 1])
        ss = stance_shifts(tag, sdir)
        bs = belief_shifts(tag, bdir)
        rel = relevance(tag, bdir)
        name_stance = ss[ss.cue_family == "implicit_demographic"]["stance_shift"].abs().mean()
        name_belief = bs[bs.cue_family == "implicit_demographic"]["belief_shift"].mean()
        rows.append({
            "model": MODEL_LABEL[tag],
            "name_decode_balacc": round(dec["bal_acc"], 2),
            "name_selectivity": round(dec["selectivity"], 2),
            "transfer_name_to_label": round(s["transfer_name_to_label_max"], 2),
            "name_belief_shift": round(name_belief, 3),
            "relevance_first_name": round(rel.get("name", np.nan), 0),
            "relevance_race": round(rel.get("race", np.nan), 0),
            "name_stance_shift_abs": round(name_stance, 3),
            "mediation_r": round(med_r, 2),
            "mediation_r_no_republican": round(med_r_norep, 2),
        })
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / "ladder_summary.csv", index=False)

    md = ["## Probe ladder — where the name cue stops mattering\n",
          "For the **name** cue (implicit demographic), one row per model across the "
          "rungs. The name is decoded and represented like an explicit label "
          "(rungs 1–2), rated far less diagnostic than the race it carries (rung 3), "
          "and barely moves the written stance (rung 4) — a use/relevance gap, not a "
          "legibility one.\n",
          "| Model | Decode bal-acc | Selectivity | Transfer name→label | Belief shift | "
          "Relevance: first name | Relevance: race | |Stance shift| | Mediation r | r (no Rep.) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        md.append(f"| {r['model']} | {r['name_decode_balacc']:.2f} | {r['name_selectivity']:.2f} | "
                  f"{r['transfer_name_to_label']:.2f} | {r['name_belief_shift']:+.3f} | "
                  f"{r['relevance_first_name']:.0f} | {r['relevance_race']:.0f} | "
                  f"{r['name_stance_shift_abs']:.3f} | {r['mediation_r']:.2f} | "
                  f"{r['mediation_r_no_republican']:.2f} |")
    md.append("\n*Decode bal-acc / selectivity: 4-way group decodability of the name at the best "
              "layer vs a shuffled-label control. Transfer: a race×gender probe trained on explicit "
              "labels, tested on names (4-way chance 0.25). Relevance: self-rated 0–100 usefulness for "
              "predicting opinion. Stance shift is mean |·| over the four name groups (full_3x). "
              "Mediation r pairs the internal Dem–Rep axis shift with the written-stance shift across "
              "all 14 cue groups; the last column removes the Republican cue, which shows the "
              "correlation is a leverage point (see the B2/B3 limitation in the text) — the internal "
              "political-axis projection is confounded by a cue-presence offset and is not shipped as "
              "a figure.*\n")
    (out_dir / "ladder_summary.md").write_text("\n".join(md))
    return t


# --------------------------------------------------------------------------- #
# Fig P1 — legibility vs use (centrepiece)
# --------------------------------------------------------------------------- #
def fig_legibility_use(pdir: Path, sdir: Path, out: Path, fmts) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    for tag in MODELS:
        s = summary(tag, pdir)
        ss = stance_shifts(tag, sdir)
        for fam, subset in FAM_SUBSET.items():
            sel = s["decodability_best"][subset]["selectivity"]
            use = ss[ss.cue_family == fam]["stance_shift"].abs().mean()
            ax.scatter(sel, use, s=120, color=FAM_COLOUR[fam], marker="o",
                       edgecolor=MODEL_COLOUR[tag], linewidth=2.2, zorder=3)

    ax.set_xlabel(r"Internal legibility  (probe selectivity: decodable group signal)",
                  fontsize=11.5)
    ax.set_ylabel(r"Behavioural use  (|written-stance shift|, cued − baseline)", fontsize=11.5)
    ax.set_ylim(bottom=-0.015)
    # the read-but-unused cluster is bottom-right: most legible, least used
    ax.annotate("most legible,\nleast used", xy=(0.758, 0.02), xytext=(0.70, 0.135),
                fontsize=9.5, color="#888888", style="italic", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=1.1))

    fam_handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=FAM_COLOUR[f],
                              label=FAM_LABEL[f]) for f in FAM_SUBSET]
    mdl_handles = [plt.Line2D([], [], marker="o", ls="", ms=10, mfc="white",
                              mec=MODEL_COLOUR[m], mew=2.2, label=MODEL_LABEL[m]) for m in MODELS]
    leg1 = ax.legend(handles=fam_handles, loc="upper right", frameon=False, fontsize=9.5,
                     title="Cue family (fill)")
    leg1.get_title().set_fontsize(9.5)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=mdl_handles, loc="center right", frameon=False, fontsize=9.5,
                     title="Model (edge)")
    leg2.get_title().set_fontsize(9.5)
    fig.tight_layout()
    _save(fig, out, "fig_p1_legibility_use", fmts)


# --------------------------------------------------------------------------- #
# Fig P2 — cross-cue transfer
# --------------------------------------------------------------------------- #
def fig_transfer(pdir: Path, out: Path, fmts) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(MODELS))[::-1]
    h = 0.34
    for i, tag in enumerate(MODELS):
        s = summary(tag, pdir)
        ax.barh(y[i] + h / 2, s["transfer_name_to_label_max"], height=h,
                color="#F0821E", edgecolor="#222", linewidth=0.5,
                label="train NAME → test label" if i == 0 else None)
        ax.barh(y[i] - h / 2, s["transfer_label_to_name_max"], height=h,
                color="#C7372F", edgecolor="#222", linewidth=0.5,
                label="train label → test NAME" if i == 0 else None)
        for val, yy in [(s["transfer_name_to_label_max"], y[i] + h / 2),
                        (s["transfer_label_to_name_max"], y[i] - h / 2)]:
            ax.text(val + 0.01, yy, f"{val:.2f}", va="center", fontsize=9)
    ax.axvline(0.25, color="#444", ls="--", lw=1.1)
    ax.text(0.25, len(MODELS) - 0.35, " chance (4-way)", color="#444", fontsize=9, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Cross-cue transfer accuracy (race × gender, max over layers)", fontsize=11.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    _save(fig, out, "fig_p2_transfer", fmts)


# --------------------------------------------------------------------------- #
# Fig P3 — direct refusal by attribute
# --------------------------------------------------------------------------- #
ATTR_ORDER = ["gender", "race", "political"]
ATTR_LABEL = {"gender": "Gender", "race": "Race", "political": "Politics"}
OUTCOME_ORDER = [("committed", "#3A6EA5", "Committed"),
                 ("committed_with_caveat", "#9FB8CF", "Committed w/ caveat"),
                 ("refused", "#B2182B", "Refused")]


def fig_refusal(bdir: Path, out: Path, fmts) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 4.2), sharey=True)
    for j, tag in enumerate(MODELS):
        ax = axes[j]
        rates = direct_rates(tag, bdir)
        piv = rates.pivot(index="attribute", columns="label", values="rate").fillna(0.0)
        x = np.arange(len(ATTR_ORDER))
        left = np.zeros(len(ATTR_ORDER))
        for key, colour, _ in OUTCOME_ORDER:
            vals = np.array([piv.loc[a][key] if (a in piv.index and key in piv.columns) else 0.0
                             for a in ATTR_ORDER])
            ax.bar(x, vals, bottom=left, color=colour, width=0.66, edgecolor="white", linewidth=0.5)
            for xi, (v, b) in enumerate(zip(vals, left)):
                if v > 0.06:
                    ax.text(xi, b + v / 2, f"{round(v*100)}%", ha="center", va="center",
                            fontsize=8.5, color="white" if colour != "#9FB8CF" else "#333")
            left += vals
        ax.set_xticks(x); ax.set_xticklabels([ATTR_LABEL[a] for a in ATTR_ORDER])
        ax.set_ylim(0, 1); ax.set_title(MODEL_LABEL[tag], fontsize=11, pad=6)
        if j == 0:
            ax.set_ylabel("Share of responses")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=c, label=l)
               for _, c, l in OUTCOME_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.2, wspace=0.08)
    _save(fig, out, "fig_p3_refusal", fmts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="results/probe_internal")
    ap.add_argument("--stance-dir", default="results/full_3x")
    ap.add_argument("--belief-dir", default="results/full")
    ap.add_argument("--figures-dir", default="figures/probe_thesis")
    ap.add_argument("--tables-dir", default="results/probe_internal")
    ap.add_argument("--format", default="both", choices=["pdf", "png", "both"])
    args = ap.parse_args()
    fmts = ["pdf", "png"] if args.format == "both" else [args.format]
    pdir, sdir, bdir = Path(args.probe_dir), Path(args.stance_dir), Path(args.belief_dir)
    fd, td = Path(args.figures_dir), Path(args.tables_dir)
    fd.mkdir(parents=True, exist_ok=True)
    td.mkdir(parents=True, exist_ok=True)

    t = ladder_table(pdir, sdir, bdir, td)
    print(t.to_string(index=False))
    fig_legibility_use(pdir, sdir, fd, fmts)
    fig_transfer(pdir, fd, fmts)
    fig_refusal(bdir, fd, fmts)
    print(f"\nWrote ladder_summary.{{csv,md}} to {td} and 3 figures to {fd} as {', '.join(fmts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
