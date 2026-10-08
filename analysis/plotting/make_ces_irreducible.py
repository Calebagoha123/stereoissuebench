#!/usr/bin/env python3
"""Appendix figure: the within-party split is not reducible to observables.

The party-levels figure sets a degenerate model stance against a non-degenerate CES
party share, and the text draws the consequence: if 62% of a group support a
proposition and the assistant supports it in every response, the other 38% receive
writing on the wrong side of their own position. The natural objection is that the
38% are a *type* -- the young, the college-educated, the secular, the coastal -- so
a responder told more about the user could place them and the mismatch is an
artefact of thin conditioning rather than of writing one stance.

This figure answers that objection with the mismatch rate itself. Per party and
issue, ``ces_within_party_predictability.py`` fits 5-fold cross-validated weighted
logistic regressions of the respondent's side on progressively larger covariate
sets, and records the weighted share written to on the wrong side under each:

    filled marker    one stance for everyone       min(share, 1 - share)
    tick             + age, education, region, religiosity  (the covariates named)
    hollow marker    + every other demographic, + pid7 and ideo5

The connector is therefore everything conditioning could ever buy a responder that
knew the respondent's full observable profile -- and it is short. What is left of
the filled marker after the connector is disagreement no identity disclosure
resolves.

Rows are sorted by the single-stance mismatch, which puts the near-even issues --
where a degenerate stance is most costly -- at the top of each panel.

Reads results/full_3x/ces_within_party_predictability.csv.
Titles are omitted deliberately (a LaTeX caption carries the takeaway).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style
from _markers import marker_ms as MS

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
_style.apply(plt)

PANELS = [("republican", "Republicans", "#B2182B"),
          ("democrat", "Democrats", "#2166AC")]

BASE_COLOUR = "#333333"     # one stance for everyone: the model's situation
GAP_COLOUR = "#B9B9B9"      # what conditioning buys
NAMED_COLOUR = "#8C8C8C"    # the four covariates the text names

FULL_MODEL = "M3_plus_political_identity"
NAMED_MODEL = "M1_named"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="results/full_3x/ces_within_party_predictability.csv")
    p.add_argument("--issues", default="data/input/issues_experiment.csv")
    p.add_argument("--out", default="figures/ces_dumbbell")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    d = pd.read_csv(args.pred)
    iss = pd.read_csv(args.issues)
    iss = iss[iss["analysis_tier"] == "main"]
    label = dict(zip(iss["ces_variable"], iss["ces_item_short"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.6))

    for ax, (party, party_label, colour) in zip(axes, PANELS):
        p = d[d["party"] == party]
        base = p.groupby("issue")["baseline_mismatch"].first()
        full = p[p["model"] == FULL_MODEL].set_index("issue")["mismatch"]
        named = p[p["model"] == NAMED_MODEL].set_index("issue")["mismatch"]
        order = base.sort_values().index.tolist()

        for i, issue in enumerate(order):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#5A5A66", alpha=0.045, zorder=0)
            b, f, n = base[issue], full[issue], named[issue]
            # connector only where conditioning actually helps; a negative reduction
            # is cross-validation noise, not a responder doing better than the label
            if b - f > 0.002:
                ax.plot([f, b], [i, i], color=GAP_COLOUR, lw=2.6,
                        solid_capstyle="round", zorder=2)
                ax.plot(f, i, marker="o", ms=MS("o", 6.4), mfc="white", mec=colour,
                        mew=1.6, ls="", zorder=4)
                ax.plot(n, i, marker="|", ms=MS("|", 7.5), color=NAMED_COLOUR,
                        mew=1.6, ls="", zorder=3)
            ax.plot(b, i, marker="o", ms=MS("o", 6.4), color=BASE_COLOUR,
                    mec="white", mew=0.8, ls="", zorder=5)

        # the panel's own headline: mean over issues, equal weight per issue
        mb, mf = base.mean(), full.reindex(order).mean()
        ax.axvline(mb, color=BASE_COLOUR, lw=1.0, ls=":", zorder=1)
        ax.axvline(mf, color=colour, lw=1.0, ls=":", zorder=1)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([label.get(i, i) for i in order], fontsize=9.5)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlim(0, 0.56)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels(["0", "10%", "20%", "30%", "40%", "50%"], fontsize=10)
        ax.text(0.0, 1.055, party_label, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=12, color=colour)
        ax.text(1.0, 1.055,
                f"mean {mb * 100:.1f}%  {_style.ARROW_R}  {mf * 100:.1f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color="#666666")

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=MS("o", 6.4), color=BASE_COLOUR,
                   mec="white", mew=0.8, label="one stance for everyone (the model)"),
        plt.Line2D([], [], marker="|", ls="", ms=MS("|", 7.5), color=NAMED_COLOUR,
                   mew=1.6, label="conditioned on age, education, region, religiosity"),
        plt.Line2D([], [], marker="o", ls="", ms=MS("o", 6.4), mfc="white",
                   mec="#555555", mew=1.6,
                   label="conditioned on every observable, incl. pid7 and ideology"),
        plt.Line2D([], [], color=GAP_COLOUR, lw=2.6, solid_capstyle="round",
                   label="what conditioning buys"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.8, bbox_to_anchor=(0.5, -0.015), columnspacing=1.6,
               handletextpad=0.5)
    fig.supxlabel("Share of the party's respondents written to on the wrong side "
                  "of their own position", fontsize=11, y=0.028)

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_ces_irreducible.{ext}", bbox_inches="tight")
    print(f"Wrote {out}/fig_ces_irreducible.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
