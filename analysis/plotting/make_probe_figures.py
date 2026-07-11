#!/usr/bin/env python3
"""Figures for the identity-cue probe study (behavioral A2/A3 + internal B1/B2/B3).

Reads, per model tag:
  results/full/belief_probe_<tag>.csv     (A2 opinion + A3 relevance)
  results/full/bert_eval_<tag>.csv        (actual written stance, of record)
  results/probe_internal/<tag>_*.csv      (decodability / transfer / mediation)

Emits into figures/probe/:
  <tag>_relevance.png        A3: perceived attribute relevance (incl. "first name")
  <tag>_belief_vs_stance.png A2: model belief about user vs stance it writes
  <tag>_transfer.png         B1: label->name decodability transfer by layer
  <tag>_mediation.png        B3: internal political-axis shift vs written stance shift
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

FAM_COLOR = {
    "explicit_political": "#1F3A93", "explicit_demographic": "#C7372F",
    "implicit_political": "#58A9DE", "implicit_demographic": "#F0821E",
}
FAM_LABEL = {
    "explicit_political": "Explicit political", "explicit_demographic": "Explicit demographic",
    "implicit_political": "Implicit political (state)", "implicit_demographic": "Implicit demographic (name)",
}


def belief_and_stance(tag: str, stance_tag: str | None = None) -> pd.DataFrame:
    stance_tag = stance_tag or tag
    b = pd.read_csv(f"results/full/belief_probe_{tag}.csv", low_memory=False)
    b["score"] = pd.to_numeric(b["parsed_score"], errors="coerce")
    op = b[b["probe_kind"].eq("opinion")].copy()
    op["sign"] = pd.to_numeric(op["liberal_sign"], errors="coerce")
    op["pred_lean"] = ((op["score"] - 50) / 50) * op["sign"]
    base = op.loc[op["cue_family"].eq("baseline"), "pred_lean"].mean()
    g = op.groupby(["cue_family", "cue_group"])["pred_lean"].mean().rename("belief_lean").reset_index()
    g["belief_shift"] = g["belief_lean"] - base

    s = pd.read_csv(f"results/full/bert_eval_{stance_tag}.csv",
                    usecols=["cue_family", "cue_group", "bert_liberal_score"], low_memory=False)
    s["bert_liberal_score"] = pd.to_numeric(s["bert_liberal_score"], errors="coerce")
    sb = s.loc[s["cue_family"].eq("baseline"), "bert_liberal_score"].mean()
    sg = s.groupby(["cue_family", "cue_group"])["bert_liberal_score"].mean().rename("stance_mean").reset_index()
    sg["stance_shift"] = sg["stance_mean"] - sb
    m = g.merge(sg, on=["cue_family", "cue_group"]).query("cue_family != 'baseline'")
    return m


def fig_relevance(tag: str, out: Path) -> None:
    b = pd.read_csv(f"results/full/belief_probe_{tag}.csv", low_memory=False)
    rel = b[b["probe_kind"].eq("relevance")].copy()
    rel["score"] = pd.to_numeric(rel["parsed_score"], errors="coerce")
    means = rel.groupby("attribute")["score"].mean().sort_values(ascending=False)
    order = means.index.tolist()
    colors = ["#1F3A93" if a == "party" else "#58A9DE" if a == "state"
              else "#F0821E" if a == "name" else "#C7372F" for a in order]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(range(len(means)), means.values, color=colors, edgecolor="#222", linewidth=0.6)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([{"party": "Political party", "state": "U.S. state", "race": "Race",
                         "gender": "Gender", "name": "First name"}.get(a, a) for a in order])
    for i, v in enumerate(means.values):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=11, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Perceived usefulness (0–100)")
    ax.set_title(f"{tag}: how predictive does the model think each attribute is?\n"
                 "Self-rated usefulness for predicting opinion, mean over 19 issues", loc="left", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def _scatter(ax, m, xcol, ycol):
    for fam in m["cue_family"].unique():
        sub = m[m["cue_family"].eq(fam)]
        ax.scatter(sub[xcol], sub[ycol], s=80, color=FAM_COLOR[fam], label=FAM_LABEL[fam],
                   edgecolor="white", linewidth=0.8, zorder=3)


def fig_belief_vs_stance(tag: str, m: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 7))
    lim = max(m["belief_shift"].abs().max(), m["stance_shift"].abs().max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "--", color="#888", lw=1, zorder=1)
    ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    _scatter(ax, m, "belief_shift", "stance_shift")
    r = np.corrcoef(m["belief_shift"], m["stance_shift"])[0, 1]
    ax.set_xlabel("Model BELIEF shift  (predicted user lean, cued − baseline)")
    ax.set_ylabel("Model STANCE shift  (written liberal score, cued − baseline)")
    ax.set_title(f"{tag}: does the model act on what it believes about the user?\n"
                 f"Each point a cue group.  r = {r:.2f}.  Below y=x line = under-acts on its belief.",
                 loc="left", fontsize=12.5)
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def fig_transfer(tag: str, out: Path) -> None:
    t = pd.read_csv(f"results/probe_internal/{tag}_cross_cue_transfer.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(t["layer"], t["label_to_name"], "-o", ms=3, color="#C7372F", label="train label → test NAME")
    ax.plot(t["layer"], t["name_to_label"], "-o", ms=3, color="#F0821E", label="train NAME → test label")
    ax.axhline(0.25, color="#888", ls="--", lw=1, label="chance (4-way)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Balanced accuracy (race × gender)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{tag}: is a name's demographic encoded like an explicit label?\n"
                 "Cross-cue probe transfer by layer — above chance = the model internally reads the name",
                 loc="left", fontsize=12.5)
    ax.legend(fontsize=9.5, loc="lower right", frameon=True)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def fig_mediation(tag: str, out: Path) -> None:
    med = pd.read_csv(f"results/probe_internal/{tag}_mediation.csv")
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    _scatter(ax, med, "proj_shift", "stance_shift")
    r = np.corrcoef(med["proj_shift"], med["stance_shift"])[0, 1]
    ax.set_xlabel("Internal political-axis shift  (Dem−Rep direction, cued − baseline)")
    ax.set_ylabel("Written stance shift  (DeBERTa liberal score, cued − baseline)")
    ax.set_title(f"{tag}: internal political position mediates written stance\n"
                 f"Each point a cue group.  r = {r:.2f}.", loc="left", fontsize=12.5)
    ax.legend(fontsize=9, loc="best", frameon=True)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llama")
    ap.add_argument("--stance-tag", default=None,
                    help="bert_eval_<stance-tag>.csv (defaults to --tag; use 'openai' for gpt)")
    ap.add_argument("--figures-dir", default="figures/probe")
    args = ap.parse_args()
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)

    m = belief_and_stance(args.tag, args.stance_tag)
    fig_relevance(args.tag, fd / f"{args.tag}_relevance.png")
    fig_belief_vs_stance(args.tag, m, fd / f"{args.tag}_belief_vs_stance.png")
    n = 2
    # Internal-probe figures only when the activation-derived CSVs exist (open models).
    if Path(f"results/probe_internal/{args.tag}_cross_cue_transfer.csv").exists():
        fig_transfer(args.tag, fd / f"{args.tag}_transfer.png")
        fig_mediation(args.tag, fd / f"{args.tag}_mediation.png")
        n = 4
    print(f"Wrote {n} figures to {fd}/{args.tag}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
