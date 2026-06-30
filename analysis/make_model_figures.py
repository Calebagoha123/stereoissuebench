#!/usr/bin/env python3
"""Per-model headline figures for the two-arm cue-steering run.

For each generation model (llama / gemma / qwen) in ``results/full/eval_*.csv``:

1. ``model_vs_ces_levels_{model}.png`` — paired bars comparing the model's mean
   liberal score under each cue against the CES weighted subgroup mean for the
   matching real-world group. Model-baseline (dashed) and CES-population (dotted)
   reference lines anchor both series. The CES columns are survey-derived and
   model-independent, so they are read straight from ``cue_ces_estimates.csv``.

2. ``stance_composition_by_cue.png`` — one 3-panel facet (a panel per model). Each
   row is a cue; the bar shows the share of responses that took the conservative /
   neutral / liberal side of the issue (colourblind-safe diverging palette), with
   refusals as a thin trailing segment.

Liberal score is per-response in {-1, 0, +1}: −1 writes the conservative side of
the issue, +1 the liberal side, 0 neutral; blank = refusal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = ["llama", "gemma", "qwen"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B", "qwen": "Qwen3.6-27B",
               "openai": "GPT-5.4-mini"}

# Which stance labels feed the figures: the Qwen 5-point judge (default) or the
# trained DeBERTa-v3 cross-encoder (bert_* columns in bert_eval_*.csv). Set in main().
STANCE_SRC = "judge"


def _src_tag() -> str:
    return "" if STANCE_SRC == "judge" else "   ·   stance: DeBERTa-v3 cross-encoder"

# Cue display order, grouped by family (the taxonomy IS the story, so we keep the
# explicit/implicit x political/demographic structure rather than sort by value).
CUE_ORDER = [
    ("baseline", "baseline", "No cue\n(baseline)"),
    ("explicit_political", "republican", "I am a Republican"),
    ("explicit_political", "independent", "I am an Independent"),
    ("explicit_political", "democrat", "I am a Democrat"),
    ("explicit_demographic", "white_man", "I am a White man"),
    ("explicit_demographic", "white_woman", "I am a White woman"),
    ("explicit_demographic", "black_man", "I am a Black man"),
    ("explicit_demographic", "black_woman", "I am a Black woman"),
    ("implicit_political", "red_state", "Resident of red state"),
    ("implicit_political", "swing_state", "Resident of swing state"),
    ("implicit_political", "blue_state", "Resident of blue state"),
    ("implicit_demographic", "white_man", "Name: white male"),
    ("implicit_demographic", "black_man", "Name: black male"),
    ("implicit_demographic", "white_woman", "Name: white female"),
    ("implicit_demographic", "black_woman", "Name: black female"),
]

GROUP_COLORS = {
    "baseline": "#8C9A9A",
    "explicit_political": "#1F3A93",
    "explicit_demographic": "#C7372F",
    "implicit_political": "#58A9DE",
    "implicit_demographic": "#F0821E",
}
GROUP_LABELS = {
    "baseline": "Baseline",
    "explicit_political": "Explicit Political",
    "explicit_demographic": "Explicit Demographic",
    "implicit_political": "Implicit Political",
    "implicit_demographic": "Implicit Demographic",
}

ISSUE_LABEL = {
    "CC25_324": "Abortion (personal choice)",
    "CC25_321a": "Ban assault rifles",
    "CC25_321b": "Easier concealed-carry",
    "CC25_321c": "Gun background checks",
    "CC25_323a": "Legal status for taxpayers",
    "CC25_323b": "More border patrols",
    "CC25_323c": "Border wall",
    "CC25_323d": "Citizenship for Dreamers",
    "CC25_321d": "+10% police",
    "CC25_321e": "−10% police",
    "CC25_326a": "EPA regulate CO₂",
    "CC25_326b": "20% renewable electricity",
    "CC25_326c": "Strengthen Clean Air/Water",
    "CC25_340d": "Proof of citizenship to vote",
    "CC25_343a": "Ban youth gender care",
    "CC25_343b": "Ban K–3 LGBT instruction",
    "CC25_343c": "Protect gender-transition access",
    "CC25_326d": "Increase fossil-fuel output",
    "CC25_326e": "Halt federal oil/gas leases",
}

# Colourblind-safe diverging palette for stance composition.
STANCE_COLORS = {
    "conservative": "#B2182B",  # red  (liberal_score = -1)
    "neutral": "#DADADA",       # grey (liberal_score =  0)
    "liberal": "#2166AC",       # blue (liberal_score = +1)
    "refusal": "#5A5A5A",       # dark grey
}


def load_model(results_dir: Path, model: str, source: str = "judge") -> pd.DataFrame:
    """Return a frame with a uniform numeric ``liberal_score`` column (+ ``eval_label``
    for the judge). For ``source='bert'`` the cross-encoder's signed score is read
    from ``bert_eval_*.csv`` (no refusals — the cross-encoder always assigns a side)."""
    if source == "bert":
        df = pd.read_csv(
            results_dir / f"bert_eval_{model}.csv",
            usecols=["cue_family", "cue_group", "bert_liberal_score", "issue_id"],
            dtype=str, low_memory=False,
        ).rename(columns={"bert_liberal_score": "liberal_score"})
    else:
        df = pd.read_csv(
            results_dir / f"eval_{model}.csv",
            usecols=["cue_family", "cue_group", "liberal_score", "eval_label", "issue_id"],
            dtype=str,
        )
    df["liberal_score"] = pd.to_numeric(df["liberal_score"], errors="coerce")
    return df


def bootstrap_mean_ci(arr: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def model_estimates(df: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    base = df.loc[df["cue_family"].eq("baseline"), "liberal_score"].to_numpy(dtype="float64")
    base = base[np.isfinite(base)]
    rows = []
    for family, group, label in CUE_ORDER:
        arr = df.loc[df["cue_family"].eq(family) & df["cue_group"].eq(group), "liberal_score"].to_numpy(dtype="float64")
        mean, lo, hi = bootstrap_mean_ci(arr, rng, n_boot)
        # Shift vs no-cue baseline; bootstrap the difference of means.
        clean = arr[np.isfinite(arr)]
        if family == "baseline" or len(clean) == 0 or len(base) == 0:
            shift_mean, shift_lo, shift_hi = 0.0, 0.0, 0.0
        else:
            g_boot = rng.choice(clean, size=(n_boot, len(clean)), replace=True).mean(axis=1)
            b_boot = rng.choice(base, size=(n_boot, len(base)), replace=True).mean(axis=1)
            diff = g_boot - b_boot
            shift_mean = float(clean.mean() - base.mean())
            shift_lo, shift_hi = (float(v) for v in np.percentile(diff, [2.5, 97.5]))
        rows.append({
            "cue_family": family, "cue_group": group, "cue_label": label,
            "model_score_mean": mean, "model_score_ci_low": lo, "model_score_ci_high": hi,
            "model_score_n": int(len(clean)),
            "model_shift_mean": shift_mean, "model_shift_ci_low": shift_lo, "model_shift_ci_high": shift_hi,
        })
    return pd.DataFrame(rows)


def stance_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Per cue: share of responses that are conservative / neutral / liberal / refusal."""
    rows = []
    for family, group, label in CUE_ORDER:
        sub = df[df["cue_family"].eq(family) & df["cue_group"].eq(group)]
        n = len(sub)
        if n == 0:
            rows.append({"cue_family": family, "cue_group": group, "cue_label": label,
                         "conservative": np.nan, "neutral": np.nan, "liberal": np.nan, "refusal": np.nan})
            continue
        refusal = sub["liberal_score"].isna().sum()
        cons = (sub["liberal_score"] == -1).sum()
        neut = (sub["liberal_score"] == 0).sum()
        lib = (sub["liberal_score"] == 1).sum()
        rows.append({
            "cue_family": family, "cue_group": group, "cue_label": label,
            "conservative": cons / n, "neutral": neut / n, "liberal": lib / n, "refusal": refusal / n,
        })
    return pd.DataFrame(rows)


def stance_shares_by_issue(df: pd.DataFrame, issue_id: str) -> pd.DataFrame:
    return stance_shares(df[df["issue_id"].eq(issue_id)])


def _clean_labels(labels: list[str]) -> list[str]:
    out = []
    for label in labels:
        label = label.replace("\n", " ")
        if label.startswith("I am"):
            label = f'"{label}."'
        label = (label.replace("Name: white male", "White-male first name")
                 .replace("Name: white female", "White-female first name")
                 .replace("Name: black male", "Black-male first name")
                 .replace("Name: black female", "Black-female first name"))
        out.append(label)
    return out


def plot_levels(est: pd.DataFrame, model: str, out_path: Path) -> None:
    x = np.arange(len(est))
    width = 0.36
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    ces_y = est["ces_score_mean"].to_numpy(dtype="float64")
    model_y = est["model_score_mean"].to_numpy(dtype="float64")
    ces_err = np.vstack([ces_y - est["ces_score_ci_low"], est["ces_score_ci_high"] - ces_y])
    model_err = np.vstack([model_y - est["model_score_ci_low"], est["model_score_ci_high"] - model_y])
    colors = [GROUP_COLORS[f] for f in est["cue_family"]]

    ax.bar(x - width / 2, ces_y, width=width, color="white", edgecolor="#2C4055", linewidth=1.3,
           yerr=ces_err, error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#2C4055"})
    ax.bar(x + width / 2, model_y, width=width, color=colors, edgecolor="#222222", linewidth=0.7,
           yerr=model_err, error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#111111"})

    baseline_model = float(est.loc[est["cue_family"].eq("baseline"), "model_score_mean"].iloc[0])
    population_ces = float(est.loc[est["cue_family"].eq("baseline"), "ces_score_mean"].iloc[0])
    ax.axhline(baseline_model, color="#8C9A9A", linestyle="--", linewidth=1.3)
    ax.axhline(population_ces, color="#2C4055", linestyle=":", linewidth=1.3)

    ax.set_title(
        f"{MODEL_LABEL[model]}: stance written under each cue vs CES subgroup opinion{_src_tag()}\n"
        "Mean liberal-score per cue, 95% bootstrap CIs  |  "
        "dashed = model no-cue baseline, dotted = CES population mean",
        loc="left", fontsize=15, pad=12,
    )
    ax.set_ylabel("Mean liberal-score  (−1 conservative … +1 liberal)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(est["cue_label"], rotation=42, ha="right", fontsize=10)
    ax.set_ylim(-0.65, 1.0)
    ax.tick_params(axis="y", labelsize=11)

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="#2C4055", label="CES weighted subgroup mean"),
        mpatches.Patch(facecolor=GROUP_COLORS["baseline"], edgecolor="#222222", label="Model no-cue baseline"),
        mpatches.Patch(facecolor=GROUP_COLORS["explicit_political"], label=GROUP_LABELS["explicit_political"]),
        mpatches.Patch(facecolor=GROUP_COLORS["explicit_demographic"], label=GROUP_LABELS["explicit_demographic"]),
        mpatches.Patch(facecolor=GROUP_COLORS["implicit_political"], label=GROUP_LABELS["implicit_political"]),
        mpatches.Patch(facecolor=GROUP_COLORS["implicit_demographic"], label=GROUP_LABELS["implicit_demographic"]),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncols=3, fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_stance_composition(shares: dict[str, pd.DataFrame], out_path: Path) -> None:
    n_cues = len(CUE_ORDER)
    y = np.arange(n_cues)[::-1]  # first cue at top
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16.5, 8.2), sharey=True)

    seg_order = ["conservative", "neutral", "liberal", "refusal"]
    for ax, model in zip(axes, MODELS):
        df = shares[model].set_index(["cue_family", "cue_group"]).reindex(
            [(f, g) for f, g, _ in CUE_ORDER]
        )
        left = np.zeros(n_cues)
        for seg in seg_order:
            vals = df[seg].to_numpy(dtype="float64")
            ax.barh(y, vals, left=left, color=STANCE_COLORS[seg], height=0.74,
                    edgecolor="white", linewidth=0.5)
            left += np.nan_to_num(vals)
        # faint separators between cue families
        boundaries = []
        prev = CUE_ORDER[0][0]
        for i, (f, _, _) in enumerate(CUE_ORDER):
            if f != prev:
                boundaries.append(i)
                prev = f
        for b in boundaries:
            ax.axhline(y[b] + 0.5, color="#cccccc", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(MODEL_LABEL[model], fontsize=13, pad=8)
        ax.set_xlabel("Share of responses", fontsize=11)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(_clean_labels([lbl for _, _, lbl in CUE_ORDER]), fontsize=10)

    legend_handles = [
        mpatches.Patch(facecolor=STANCE_COLORS["conservative"], label="Conservative side"),
        mpatches.Patch(facecolor=STANCE_COLORS["neutral"], label="Neutral / balanced"),
        mpatches.Patch(facecolor=STANCE_COLORS["liberal"], label="Liberal side"),
        mpatches.Patch(facecolor=STANCE_COLORS["refusal"], label="Refusal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"How each cue reshapes the stance the model writes{_src_tag()}\n"
        "Share of responses taking the conservative / neutral / liberal side of the issue, by cue",
        x=0.012, ha="left", fontsize=16, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_did(est: pd.DataFrame, model: str, out_path: Path) -> None:
    x = np.arange(len(est))
    width = 0.36
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    ces_y = est["ces_shift_mean"].to_numpy(dtype="float64")
    model_y = est["model_shift_mean"].to_numpy(dtype="float64")
    ces_err = np.vstack([ces_y - est["ces_shift_ci_low"], est["ces_shift_ci_high"] - ces_y])
    model_err = np.vstack([model_y - est["model_shift_ci_low"], est["model_shift_ci_high"] - model_y])
    colors = [GROUP_COLORS[f] for f in est["cue_family"]]

    ax.bar(x - width / 2, ces_y, width=width, color="white", edgecolor="#2C4055", linewidth=1.3,
           yerr=ces_err, error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#2C4055"})
    ax.bar(x + width / 2, model_y, width=width, color=colors, edgecolor="#222222", linewidth=0.7,
           yerr=model_err, error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#111111"})

    ax.set_title(
        f"{MODEL_LABEL[model]}: cue effect as difference-in-differences vs CES{_src_tag()}\n"
        "95% bootstrap CIs.  Model: cued − baseline.   CES: subgroup − population.",
        loc="left", fontsize=15, pad=12,
    )
    ax.set_ylabel("Mean liberal-score shift", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(est["cue_label"], rotation=42, ha="right", fontsize=10)
    ax.set_ylim(-0.95, 0.95)
    ax.tick_params(axis="y", labelsize=11)
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="#2C4055", label="CES shift (subgroup − population)"),
        mpatches.Patch(facecolor=GROUP_COLORS["explicit_political"], label=GROUP_LABELS["explicit_political"]),
        mpatches.Patch(facecolor=GROUP_COLORS["explicit_demographic"], label=GROUP_LABELS["explicit_demographic"]),
        mpatches.Patch(facecolor=GROUP_COLORS["implicit_political"], label=GROUP_LABELS["implicit_political"]),
        mpatches.Patch(facecolor=GROUP_COLORS["implicit_demographic"], label=GROUP_LABELS["implicit_demographic"]),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncols=2, fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_forest(est: pd.DataFrame, model: str, out_path: Path) -> None:
    """Dot + 95% CI of each cue's shift from the model's own no-cue baseline (0)."""
    effects = est[~est["cue_family"].eq("baseline")].copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(effects))
    fig, ax = plt.subplots(figsize=(12.5, 9.2))

    band_colors = {
        "explicit_political": "#F4F6FB", "explicit_demographic": "#FCF5F3",
        "implicit_political": "#F4FAFD", "implicit_demographic": "#FCF6EF",
    }
    for family in effects["cue_family"].unique():
        idx = np.where(effects["cue_family"].to_numpy() == family)[0]
        ax.axhspan(idx.min() - 0.5, idx.max() + 0.5, color=band_colors[family], zorder=0)
        ax.text(0.62, idx.mean(), GROUP_LABELS[family], color=GROUP_COLORS[family],
                fontsize=13, weight="bold", va="center")

    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.7, alpha=0.8)
    colors = [GROUP_COLORS[f] for f in effects["cue_family"]]
    xv = effects["model_shift_mean"].to_numpy(dtype="float64")
    xerr = np.vstack([xv - effects["model_shift_ci_low"], effects["model_shift_ci_high"] - xv])
    for i, color in enumerate(colors):
        ax.errorbar(xv[i], y[i], xerr=np.array([[xerr[0, i]], [xerr[1, i]]]), fmt="none",
                    ecolor=color, capsize=4, linewidth=1.4, zorder=2)
    ax.scatter(xv, y, s=70, color=colors, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(_clean_labels(list(effects["cue_label"])), fontsize=11)
    ax.set_xlim(-1.0, 0.95)
    ax.set_xlabel(
        "Cue effect on liberal-score (vs no-cue baseline)\n"
        "Negative = output shifts conservative   |   Positive = output shifts liberal",
        fontsize=12,
    )
    total_n = int(est["model_score_n"].sum())
    ax.set_title(
        f"{MODEL_LABEL[model]} — cue effect on political stance in writing assistance{_src_tag()}\n"
        f"Group-mean shift from no-cue baseline, 95% bootstrap CIs, n={total_n:,} scored generations",
        loc="left", fontsize=16, pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_stance_by_issue(df: pd.DataFrame, model: str, out_path: Path) -> None:
    issues = list(ISSUE_LABEL.keys())
    ncols, nrows = 4, 5
    n_cues = len(CUE_ORDER)
    y = np.arange(n_cues)[::-1]
    seg_order = ["conservative", "neutral", "liberal", "refusal"]

    # family separators (positions where family changes)
    boundaries = [i for i in range(1, n_cues) if CUE_ORDER[i][0] != CUE_ORDER[i - 1][0]]

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 23), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, issue in zip(axes, issues):
        sh = stance_shares_by_issue(df, issue).set_index(["cue_family", "cue_group"]).reindex(
            [(f, g) for f, g, _ in CUE_ORDER]
        )
        left = np.zeros(n_cues)
        for seg in seg_order:
            vals = sh[seg].to_numpy(dtype="float64")
            ax.barh(y, vals, left=left, color=STANCE_COLORS[seg], height=0.78,
                    edgecolor="white", linewidth=0.4)
            left += np.nan_to_num(vals)
        for b in boundaries:
            ax.axhline(y[b] + 0.5, color="#cccccc", linewidth=0.7)
        ax.set_xlim(0, 1)
        ax.set_title(ISSUE_LABEL[issue], fontsize=11)
        ax.tick_params(axis="y", length=0)
    # blank out unused cells
    for ax in axes[len(issues):]:
        ax.axis("off")
    # y labels only on leftmost column
    cue_labels = _clean_labels([lbl for _, _, lbl in CUE_ORDER])
    for r in range(nrows):
        ax = axes[r * ncols]
        ax.set_yticks(y)
        ax.set_yticklabels(cue_labels, fontsize=9)
    for c in range(ncols):
        axes[(nrows - 1) * ncols + c].set_xlabel("Share of responses", fontsize=10)

    legend_handles = [
        mpatches.Patch(facecolor=STANCE_COLORS["conservative"], label="Conservative side"),
        mpatches.Patch(facecolor=STANCE_COLORS["neutral"], label="Neutral / balanced"),
        mpatches.Patch(facecolor=STANCE_COLORS["liberal"], label="Liberal side"),
        mpatches.Patch(facecolor=STANCE_COLORS["refusal"], label="Refusal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=4, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"{MODEL_LABEL[model]}: stance written under each cue, broken out by issue{_src_tag()}\n"
        "Share of responses on the conservative / neutral / liberal side, per cue (rows) within each issue (panels)",
        x=0.012, ha="left", fontsize=17, y=0.997,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Stance-judge 5-point scale (1 = only pro / supports the issue ... 5 = only con).
LABEL_COLS = ["1", "2", "3", "4", "5", "refusal"]
LABEL_HEADER = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "refusal": "R"}
LABEL_HEAD_BG = {
    "1": "#3C7A3F", "2": "#8DC891", "3": "#F6D365",
    "4": "#E68A8A", "5": "#CC2222", "refusal": "#B3B3B3",
}
LABEL_HEAD_FG = {"1": "white", "2": "black", "3": "black", "4": "black", "5": "white", "refusal": "black"}


# BERT cross-encoder produces a 3-way side only (no 1-5 Likert, no refusal), shown
# in liberal orientation to match the stance-composition figures.
BERT_LABEL_COLS = ["liberal", "neutral", "conservative"]
BERT_LABEL_HEADER = {"liberal": "Lib", "neutral": "Neu", "conservative": "Con"}
BERT_HEAD_BG = {"liberal": "#2166AC", "neutral": "#DADADA", "conservative": "#B2182B"}
BERT_HEAD_FG = {"liberal": "white", "neutral": "black", "conservative": "white"}


def label_counts(df: pd.DataFrame, source: str = "judge") -> dict[str, int]:
    if source == "bert":
        ls = df["liberal_score"]
        counts = {"liberal": int((ls == 1).sum()), "neutral": int((ls == 0).sum()),
                  "conservative": int((ls == -1).sum())}
    else:
        vc = df["eval_label"].value_counts()
        counts = {k: int(vc.get(k, 0)) for k in LABEL_COLS}
    counts["total"] = int(len(df))
    return counts


def plot_label_table(counts: dict[str, dict[str, int]], out_path: Path, source: str = "judge") -> None:
    is_bert = source == "bert"
    label_cols = BERT_LABEL_COLS if is_bert else LABEL_COLS
    head_txt = BERT_LABEL_HEADER if is_bert else LABEL_HEADER
    head_bg = BERT_HEAD_BG if is_bert else LABEL_HEAD_BG
    head_fg = BERT_HEAD_FG if is_bert else LABEL_HEAD_FG
    footnote = (
        "Cross-encoder stance (DeBERTa-v3): Lib = liberal side · Neu = neutral · "
        "Con = conservative side.   Counts = scored generations."
        if is_bert else
        "Stance-judge labels: 1 = only pro · 2 = mostly pro · 3 = neutral · "
        "4 = mostly con · 5 = only con · R = refusal.   Counts = scored generations."
    )

    rows = MODELS
    value_cols = label_cols + ["total"]
    x_centers = np.linspace(0.40, 0.96, len(value_cols))
    x_model = 0.015
    y_header = 0.86
    row_h = 0.74 / max(len(rows), 1)
    y_rows = [y_header - 0.13 - i * row_h for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(8.6, 0.9 + 0.5 * len(rows)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # alternating row backgrounds
    for i, y in enumerate(y_rows):
        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle((0, y - row_h / 2), 1, row_h,
                                            facecolor="#F0F0F0", edgecolor="none", zorder=0))

    # header
    ax.text(x_model, y_header, "Model", fontsize=13, weight="bold", va="center", ha="left")
    cell_w, cell_h = 0.052, 0.075
    for c, x in zip(value_cols, x_centers):
        if c in head_bg:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - cell_w / 2, y_header - cell_h / 2), cell_w, cell_h,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=head_bg[c], edgecolor="none", zorder=1))
            ax.text(x, y_header, head_txt[c], fontsize=12.5, weight="bold",
                    color=head_fg[c], va="center", ha="center", zorder=2)
        else:
            ax.text(x, y_header, "Total", fontsize=13, weight="bold", va="center", ha="center")

    # rule under header
    ax.plot([0, 1], [y_header - 0.085, y_header - 0.085], color="#333", lw=1.4)

    # rows
    for y, model in zip(y_rows, rows):
        ax.text(x_model, y, MODEL_LABEL[model], fontsize=12.5, va="center", ha="left")
        for c, x in zip(value_cols, x_centers):
            val = counts[model][c]
            weight = "bold" if c == "total" else "normal"
            ax.text(x, y, f"{val:,}", fontsize=12, va="center", ha="center", weight=weight)

    ax.plot([0, 1], [y_rows[-1] - row_h / 2, y_rows[-1] - row_h / 2], color="#333", lw=1.4)
    fig.text(0.015, 0.04, footnote, fontsize=9, color="#444")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global STANCE_SRC, MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full")
    ap.add_argument("--ces-estimates", default="results/main/cue_ces_estimates.csv")
    ap.add_argument("--figures-dir", default=None,
                    help="default: figures/full (judge) or figures/full_bert (bert)")
    ap.add_argument("--label-source", choices=["judge", "bert"], default="judge",
                    help="stance labels: Qwen 5-point judge, or the trained DeBERTa cross-encoder")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated generation models to plot (need matching eval files)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()

    MODELS = [m.strip() for m in args.models.split(",") if m.strip()]
    STANCE_SRC = args.label_source
    figures_dir = args.figures_dir or ("figures/full_bert" if STANCE_SRC == "bert" else "figures/full")
    rd, fd = Path(args.results_dir), Path(figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ces = pd.read_csv(args.ces_estimates)
    ces_cols = ["cue_family", "cue_group", "ces_score_mean", "ces_score_ci_low",
                "ces_score_ci_high", "ces_population_mean", "ces_shift_mean",
                "ces_shift_ci_low", "ces_shift_ci_high"]
    ces = ces[ces_cols]

    print(f"Label source: {STANCE_SRC}  ->  {fd}")
    shares = {}
    counts = {}
    for model in MODELS:
        df = load_model(rd, model, STANCE_SRC)
        est = model_estimates(df, rng, args.bootstrap)
        est = est.merge(ces, on=["cue_family", "cue_group"], how="left")
        plot_levels(est, model, fd / f"model_vs_ces_levels_{model}.png")
        plot_did(est, model, fd / f"model_vs_ces_did_{model}.png")
        plot_forest(est, model, fd / f"model_cue_effects_{model}.png")
        plot_stance_by_issue(df, model, fd / f"stance_composition_by_issue_{model}.png")
        shares[model] = stance_shares(df)
        counts[model] = label_counts(df, STANCE_SRC)
        print(f"Wrote levels/did/forest/by-issue for {model}")

    plot_stance_composition(shares, fd / "stance_composition_by_cue.png")
    plot_label_table(counts, fd / "stance_label_counts.png", STANCE_SRC)
    print(f"Wrote {fd / 'stance_composition_by_cue.png'}")
    print(f"Wrote {fd / 'stance_label_counts.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
