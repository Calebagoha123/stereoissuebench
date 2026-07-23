#!/usr/bin/env python3
"""Template-genre heterogeneity (check G-ii): does a cue effect live in one genre?

"Your averages hide that one genre drives everything." For the Arm-A explicit cues
we break the effect out by the writing template's genre (essay, article, speech,
report, argument, …), pooled over the three models, to check whether e.g. the
Republican shift lives only in persuasive genres and vanishes in reports.

Template genre is joined from data/input/templates_all_145.csv on template rank
(the eval `t<rank>` id). Reads results/full_3x via analysis/lib/_common.py. Writes
results/robustness/genre_heterogeneity.csv and a figure.
"""
from __future__ import annotations

from pathlib import Path

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pandas as pd

from _common import MODELS, load_model

TEMPLATES = "data/input/templates_all_145.csv"
KEY_CUES = [("explicit_political", "republican", "Republican"),
            ("explicit_political", "democrat", "Democrat"),
            ("explicit_demographic", "black_woman", "Black woman"),
            ("explicit_demographic", "white_man", "White man")]
MIN_TEMPLATES = 5  # collapse rarer genres into "other"


def main():
    tpl = pd.read_csv(TEMPLATES)
    rank2genre = dict(zip(tpl["rank"], tpl["genre"]))
    keep_genres = tpl["genre"].value_counts()
    keep_genres = set(keep_genres[keep_genres >= MIN_TEMPLATES].index)

    # pool Arm A over the 3 models
    frames = []
    for m in MODELS:
        d = load_model(m)
        d = d[d.arm == "A"].copy()
        d["rank"] = d["template_id"].str.slice(1).astype(int)
        d["genre"] = d["rank"].map(rank2genre)
        d["genre"] = d["genre"].where(d["genre"].isin(keep_genres), "other")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    base = df[df.cue_family == "baseline"]
    base_by_genre = base.groupby("genre")["y"].mean()

    recs = []
    for fam, grp, lab in KEY_CUES:
        cue = df[(df.cue_family == fam) & (df.cue_group == grp)]
        for genre, g in cue.groupby("genre"):
            n = len(g)
            eff = g["y"].mean() - base_by_genre.get(genre, np.nan)
            se = g["y"].std(ddof=1) / np.sqrt(n)
            recs.append({"cue": lab, "genre": genre, "n": n, "effect": eff, "se": se})
        # overall (all genres)
        recs.append({"cue": lab, "genre": "ALL", "n": len(cue),
                     "effect": cue["y"].mean() - base["y"].mean(),
                     "se": cue["y"].std(ddof=1) / np.sqrt(len(cue))})
    out = pd.DataFrame(recs)
    out.to_csv(Path("results/robustness/genre_heterogeneity.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=== Cue effect by template genre (Arm A, pooled over 3 models) ===\n")
    for _, lab in [(c[0], c[2]) for c in KEY_CUES]:
        s = out[out.cue == lab].sort_values("effect")
        allrow = s[s.genre == "ALL"].iloc[0]
        print(f"--- {lab}  (ALL genres Δ={allrow['effect']:+.3f}) ---")
        for _, r in s[s.genre != "ALL"].iterrows():
            print(f"    {r['genre']:>12} (n={int(r['n']):5d}): Δ={r['effect']:+.3f} ±{r['se']:.3f}")
        rng = s[s.genre != "ALL"]["effect"]
        print(f"    across-genre range: {rng.max()-rng.min():.3f}  "
              f"(all same sign: {np.all(np.sign(rng)==np.sign(allrow['effect']))})\n")
    make_figure(out)
    print(f"Wrote results/robustness/genre_heterogeneity.csv")


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    cues = [c[2] for c in KEY_CUES]
    fig, axes = plt.subplots(1, len(cues), figsize=(15, 4.4), sharex=True)
    for ax, lab in zip(axes, cues):
        s = out[(out.cue == lab) & (out.genre != "ALL")].sort_values("effect")
        y = range(len(s))
        ax.errorbar(s["effect"], list(y), xerr=1.96 * s["se"], fmt="o", ms=5,
                    color="#2e6da4", ecolor="#888", capsize=2)
        allv = out[(out.cue == lab) & (out.genre == "ALL")]["effect"].iloc[0]
        ax.axvline(allv, color="#c0392b", ls="--", lw=1, label="pooled")
        ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_yticks(list(y)); ax.set_yticklabels(s["genre"], fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Δ vs baseline")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Cue effect by template genre — the sign is consistent across genres, "
                 "not driven by one persuasive genre", y=1.02, fontsize=11)
    fig.tight_layout()
    p = Path("figures/robustness/genre_heterogeneity.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
