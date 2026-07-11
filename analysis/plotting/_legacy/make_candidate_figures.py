#!/usr/bin/env python3
"""Candidate CROSS-MODEL figures for the cue-steering run.

Title-free (a LaTeX caption carries the takeaway). All effect estimates carry
95% CIs, clustered on CES issue. Classifier of record: DeBERTa
``bert_liberal_score`` in {-1,0,+1} (+ = wrote the liberal side); the BERT eval
also unlocks the 4th model, gpt-5.4-mini, absent from the Qwen-judged eval_*.csv.

Outputs → figures/candidates/:
  B_political_asymmetry.png   baseline lean + Dem/Rep pull, CIs; only Rep flips
  E_cue_family_dumbbells.png  B x4 — one 2x2 panel per cue family, own scale, CIs
  G_ces_calibration.png       model shift vs CES real-group shift (stereotype vs
                              personalisation): y=x = calibrated, above = amplify
  F_issue_heatmap.png         per-issue Republican-cue shift x model, heatmap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
    "font.family": "DejaVu Sans",
})

MODELS = ["llama", "gemma", "qwen", "openai"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B",
               "qwen": "Qwen3-27B", "openai": "GPT-5.4-mini"}
MODEL_COLOUR = {"llama": "#0077BB", "gemma": "#009988",
                "qwen": "#EE7733", "openai": "#CC3311"}
SCORE = "bert_liberal_score"
LIB, CON, BASE_C, MID = "#2166AC", "#B2182B", "#444444", "#9aa0a6"
Z = 1.96


def load(results_dir: Path) -> dict[str, pd.DataFrame]:
    cols = ["arm", "cue_condition", "cue_group", "issue_id", "stance_target", SCORE]
    return {m: pd.read_csv(results_dir / f"bert_eval_{m}.csv", usecols=cols,
                           low_memory=False) for m in MODELS}


# --- estimators (95% CI clustered on issue) --------------------------------
def _issue_means(df: pd.DataFrame, mask) -> pd.Series:
    return df[mask].groupby("issue_id")[SCORE].mean()


def mean_ci(df, mask) -> tuple[float, float]:
    im = _issue_means(df, mask)
    return df[mask][SCORE].mean(), Z * im.std(ddof=1) / np.sqrt(len(im))


def shift_ci(df, cued_mask, base_mask) -> tuple[float, float]:
    d = (_issue_means(df, cued_mask) - _issue_means(df, base_mask)).dropna()
    pooled = df[cued_mask][SCORE].mean() - df[base_mask][SCORE].mean()
    return pooled, Z * d.std(ddof=1) / np.sqrt(len(d))


def base_mask(df):
    return (df.arm == "A") & (df.cue_condition == "baseline")


def cue_mask(df, family, group):
    if family.startswith("explicit"):
        return (df.arm == "A") & (df.cue_condition == f"{family}_{group}")
    return (df.arm == "B") & (df.cue_group == group)


# ----------------------------------------------------------------------------
# B. Political asymmetry (main-experiment pick), with CIs
# ----------------------------------------------------------------------------
def fig_political_asymmetry(data, out: Path):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ys = np.arange(len(MODELS))[::-1]
    for y, m in zip(ys, MODELS):
        df = data[m]
        bm = base_mask(df)
        base, base_e = mean_ci(df, bm)
        dem, dem_e = mean_ci(df, (df.arm == "A") & (df.cue_condition == "explicit_political_democrat"))
        rep, rep_e = mean_ci(df, (df.arm == "A") & (df.cue_condition == "explicit_political_republican"))
        ax.plot([rep, dem], [y, y], color="#cccccc", lw=2, zorder=1)
        ax.errorbar([rep], [y], xerr=[[rep_e], [rep_e]], fmt="o", ms=11, color=CON,
                    ecolor=CON, elinewidth=1.4, capsize=3, zorder=3, mec="white", mew=1.1)
        ax.errorbar([dem], [y], xerr=[[dem_e], [dem_e]], fmt="o", ms=11, color=LIB,
                    ecolor=LIB, elinewidth=1.4, capsize=3, zorder=3, mec="white", mew=1.1)
        ax.errorbar([base], [y], xerr=[[base_e], [base_e]], fmt="D", ms=8, color=BASE_C,
                    ecolor=BASE_C, elinewidth=1.2, capsize=3, zorder=4, mec="white", mew=1)
        ax.annotate(f"{rep:+.2f}", (rep, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8.5, color=CON, fontweight="bold")
        ax.annotate(f"{dem:+.2f}", (dem, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8.5, color=LIB, fontweight="bold")
    ax.axvline(0, color="#333", lw=1.1, zorder=2)
    ax.set_ylim(-0.85, len(MODELS) - 0.35)
    ax.text(0, -0.7, "neutral", ha="center", va="center", fontsize=8.5, color="#333",
            style="italic", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_yticks(ys)
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_xlabel("mean liberal score  (− conservative side  ·  + liberal side)")
    ax.set_xlim(-0.55, 0.85)
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], marker="D", ls="", color=BASE_C, label="baseline (no cue)"),
        plt.Line2D([], [], marker="o", ls="", color=LIB, label='"user is a Democrat"'),
        plt.Line2D([], [], marker="o", ls="", color=CON, label='"user is a Republican"'),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.34))
    fig.tight_layout()
    fig.savefig(out / "B_political_asymmetry.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# E. B x4 — 2x2, one panel per cue family, own scale, CIs
# ----------------------------------------------------------------------------
def fig_cue_family_dumbbells(data, out: Path):
    # per family: (title, conservative-pole, liberal-pole, [mid dots], endpoint hint)
    families = [
        ("Explicit political", "explicit_political", "republican", "democrat",
         [("independent", "Independent")], "◀ Republican   Democrat ▶"),
        ("Explicit demographic  (label)", "explicit_demographic", "white_man", "black_woman",
         [("white_woman", "White woman"), ("black_man", "Black man")],
         "◀ “As a White man”   “As a Black woman” ▶"),
        ("Implicit political  (US state)", "implicit_political", "red_state", "blue_state",
         [("swing_state", "Swing state")], "◀ Red state   Blue state ▶"),
        ("Implicit demographic  (name)", "implicit_demographic", "white_man", "black_woman",
         [("white_woman", "White-woman name"), ("black_man", "Black-man name")],
         "◀ White-man name   Black-woman name ▶"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ys = np.arange(len(MODELS))[::-1]
    for ax, (title, fam, con_g, lib_g, mids, hint) in zip(axes.ravel(), families):
        for y, m in zip(ys, MODELS):
            df = data[m]
            base, base_e = mean_ci(df, base_mask(df))
            con, con_e = mean_ci(df, cue_mask(df, fam, con_g))
            lib, lib_e = mean_ci(df, cue_mask(df, fam, lib_g))
            midvals = [mean_ci(df, cue_mask(df, fam, g))[0] for g, _ in mids]
            pts = [con, lib] + midvals
            ax.plot([min(pts), max(pts)], [y, y], color="#d3d3d3", lw=2.2, zorder=1)
            for mv in midvals:
                ax.scatter(mv, y, s=34, color=MID, zorder=2, edgecolor="white", lw=0.6)
            ax.errorbar(con, y, xerr=con_e, fmt="o", ms=9, color=CON, ecolor=CON,
                        elinewidth=1.2, capsize=2.5, zorder=3, mec="white", mew=1)
            ax.errorbar(lib, y, xerr=lib_e, fmt="o", ms=9, color=LIB, ecolor=LIB,
                        elinewidth=1.2, capsize=2.5, zorder=3, mec="white", mew=1)
            ax.errorbar(base, y, xerr=base_e, fmt="D", ms=7, color=BASE_C, ecolor=BASE_C,
                        elinewidth=1, capsize=2.5, zorder=4, mec="white", mew=0.8)
        ax.axvline(0, color="#333", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16)
        ax.text(0.0, 1.13, title, transform=ax.transAxes, fontsize=11.5,
                fontweight="bold", va="bottom", ha="left")
        ax.text(0.0, 1.04, hint, transform=ax.transAxes, fontsize=8.5, color="#666",
                va="bottom", ha="left", style="italic")
    for ax in axes[1]:
        ax.set_xlabel("mean liberal score  (−  conservative  ·  liberal  +)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=CON, label="conservative-pole cue"),
        plt.Line2D([], [], marker="D", ls="", color=BASE_C, label="baseline (no cue)"),
        plt.Line2D([], [], marker="o", ls="", color=MID, label="intermediate cue"),
        plt.Line2D([], [], marker="o", ls="", color=LIB, label="liberal-pole cue"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "E_cue_family_dumbbells.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# G. CES calibration — stereotyping vs personalisation
# ----------------------------------------------------------------------------
def fig_ces_calibration(data, ces_table: Path, out: Path):
    ces = pd.read_csv(ces_table)
    ces = ces[ces.cue_group != "baseline"]
    fam_panels = [
        ("explicit_political", "Explicit political"),
        ("explicit_demographic", "Explicit demographic (label)"),
        ("implicit_political", "Implicit political (US state)"),
        ("implicit_demographic", "Implicit demographic (name)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    for ax, (fam, title) in zip(axes.ravel(), fam_panels):
        sub = ces[ces.cue_family == fam]
        xs_all, ys_all = [], []
        for _, r in sub.iterrows():
            cx = r.ces_shift_mean
            cx_e = [[cx - r.ces_shift_ci_low], [r.ces_shift_ci_high - cx]]
            # CES uncertainty (shared across models) as a faint vertical band
            ax.axvspan(r.ces_shift_ci_low, r.ces_shift_ci_high, color="#eeeeee", zorder=0)
            for m in MODELS:
                df = data[m]
                sh, e = shift_ci(df, cue_mask(df, fam, r.cue_group), base_mask(df))
                ax.errorbar(cx, sh, yerr=e, fmt="o", ms=7,
                            color=MODEL_COLOUR[m], ecolor=MODEL_COLOUR[m],
                            elinewidth=1.1, capsize=0, alpha=0.85, zorder=3,
                            mec="white", mew=0.7)
                xs_all.append(cx); ys_all.append(sh)
        lim = max(0.08, np.nanmax(np.abs(xs_all + ys_all)) * 1.15)
        ax.plot([-lim, lim], [-lim, lim], color="#555", ls="--", lw=1, zorder=1)
        ax.axhline(0, color="#bbb", lw=0.8, zorder=0)
        ax.axvline(0, color="#bbb", lw=0.8, zorder=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.text(0.03, 0.95, title, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left")
        # region hints (only on first panel to avoid clutter)
        if fam == "explicit_political":
            ax.text(0.96, 0.60, "amplifies\n(stereotype)", transform=ax.transAxes,
                    fontsize=8, color="#777", ha="right", va="center", style="italic")
            ax.text(0.60, 0.04, "compresses\n(under-personalises)", transform=ax.transAxes,
                    fontsize=8, color="#777", ha="center", va="bottom", style="italic")
    for ax in axes[1]:
        ax.set_xlabel("CES real-group stance shift  (ground truth)")
    for ax in axes[:, 0]:
        ax.set_ylabel("model stance shift  (cued − baseline)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOUR[m],
                          label=MODEL_LABEL[m]) for m in MODELS]
    handles.append(plt.Line2D([], [], ls="--", color="#555", label="y = x (calibrated)"))
    fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "G_ces_calibration.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# H. Side-taking vs CES — does the model match the real group's opinion mix?
# ----------------------------------------------------------------------------
def _p_lib_side_ci(df, mask) -> tuple[float, float]:
    """Share on the liberal side among responses that take a side (drop neutral),
    matching CES forced-choice; 95% CI clustered on issue."""
    sub = df[mask]
    def pf(x):
        lib = (x == 1).sum(); con = (x == -1).sum()
        return np.nan if (lib + con) == 0 else lib / (lib + con)
    per = sub.groupby("issue_id")[SCORE].apply(pf).dropna()
    lib = int((sub[SCORE] == 1).sum()); con = int((sub[SCORE] == -1).sum())
    pooled = lib / (lib + con) if (lib + con) else np.nan
    return pooled, Z * per.std(ddof=1) / np.sqrt(len(per))


def fig_side_vs_ces(data, ces_table: Path, out: Path):
    ces = pd.read_csv(ces_table).set_index(["cue_family", "cue_group"])
    def ces_plib(fam, g):
        return (ces.loc[(fam, g), "ces_score_mean"] + 1) / 2
    groups = [  # (header, [(family, group, label)])
        ("", [("baseline", "baseline", "No cue (population)")]),
        ("Explicit political", [
            ("explicit_political", "democrat", "Democrat"),
            ("explicit_political", "independent", "Independent"),
            ("explicit_political", "republican", "Republican")]),
        ("Explicit demographic (label)", [
            ("explicit_demographic", "black_woman", "Black woman"),
            ("explicit_demographic", "black_man", "Black man"),
            ("explicit_demographic", "white_woman", "White woman"),
            ("explicit_demographic", "white_man", "White man")]),
        ("Implicit political (US state)", [
            ("implicit_political", "blue_state", "Blue state"),
            ("implicit_political", "swing_state", "Swing state"),
            ("implicit_political", "red_state", "Red state")]),
    ]
    rows, headers, y = [], [], 0.0
    for hdr, items in groups:
        if hdr:
            headers.append((y - 0.85, hdr));
        for fam, g, lbl in items:
            rows.append((y, fam, g, lbl)); y += 1.0
        y += 1.2

    fig, ax = plt.subplots(figsize=(10, 8.4))
    blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yy, fam, g, lbl in rows:
        ces_v = ces_plib(fam, g)
        mvals = []
        for m in MODELS:
            df = data[m]
            p, e = _p_lib_side_ci(df, cue_mask(df, fam, g))
            mvals.append((m, p, e))
        # gap line from CES to model spread
        ax.plot([ces_v, np.mean([p for _, p, _ in mvals])], [yy, yy],
                color="#dddddd", lw=6, zorder=0, solid_capstyle="round")
        ax.scatter([ces_v], [yy], marker="*", s=260, color="#111", zorder=4,
                   edgecolor="white", lw=0.8)
        for m, p, e in mvals:
            ax.errorbar(p, yy, xerr=e, fmt="o", ms=8, color=MODEL_COLOUR[m],
                        ecolor=MODEL_COLOUR[m], elinewidth=1.2, capsize=0,
                        zorder=3, mec="white", mew=0.8)
    for hy, hdr in headers:
        ax.text(0.0, hy, hdr, transform=blend, fontsize=10.5, fontweight="bold",
                va="center", ha="left")
    ax.axvline(0.5, color="#333", lw=1, ls=":", zorder=1)
    top = min(r[0] for r in rows) - 1.0
    ax.text(0.5, top, "← more conservative    balanced    more liberal →",
            ha="center", va="center", fontsize=8.5, color="#333", style="italic")
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[3] for r in rows], fontsize=9.5)
    ax.set_ylim(max(r[0] for r in rows) + 0.8, top - 0.4)
    ax.set_xlim(0.12, 1.02)
    ax.set_xlabel("share on the LIBERAL side, among responses that take a side\n"
                  "(model when cued  vs  the real CES group)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="*", ls="", color="#111", ms=14,
                          label="CES real group")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOUR[m],
                           label=MODEL_LABEL[m]) for m in MODELS]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=5,
              loc="lower center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out / "H_side_taking_vs_ces.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full")
    ap.add_argument("--figures-dir", default="figures/candidates")
    ap.add_argument("--ces-table", default="results/full/rq2_bert_vs_ces.csv")
    args = ap.parse_args()
    rd, fd = Path(args.results_dir), Path(args.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    data = load(rd)
    fig_political_asymmetry(data, fd)
    fig_cue_family_dumbbells(data, fd)
    fig_ces_calibration(data, Path(args.ces_table), fd)
    fig_side_vs_ces(data, Path(args.ces_table), fd)
    print(f"Wrote 4 candidate figures to {fd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
