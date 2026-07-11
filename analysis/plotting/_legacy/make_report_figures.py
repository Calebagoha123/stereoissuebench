#!/usr/bin/env python3
"""Make the three report figures for the cue-personalization experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluated", default="data/processed/evaluated_with_effects.csv")
    parser.add_argument("--ground", default="data/reference/ces_ground_truth_template.csv")
    parser.add_argument(
        "--ces-dta",
        default="../CES/CES25_Common.dta",
        help="Respondent-level CES Stata file. This file is not committed.",
    )
    parser.add_argument("--weight-col", default="commonweight")
    parser.add_argument("--figures-dir", default="figures/main")
    parser.add_argument("--results-dir", default="results/main")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def libify(values: pd.Series, sign: int, item: str) -> pd.Series:
    values_num = pd.to_numeric(values, errors="coerce")
    if item == "CC25_324":
        mapped = (values_num - 2.5) / 1.5
    else:
        mapped = np.where(values_num == 1, 1.0, np.where(values_num == 2, -1.0, np.nan))
    return pd.Series(mapped, index=values.index, dtype="float64") * sign


def weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(dtype="float64")
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


def subgroup_masks(df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    race = df["race"].map({1: "white", 2: "black"})
    gender = df["gender4"].map({1: "man", 2: "woman"})
    race_gender = race.fillna("") + "_" + gender.fillna("")
    race_gender[race.isna() | gender.isna()] = np.nan

    pid3 = df["pid3"].map({1: "democrat", 2: "republican", 3: "independent"})
    blue_states = {6, 25, 36}
    red_states = {1, 40, 48}
    swing_states = {13, 42, 55}
    state_part = df["inputstate"].apply(
        lambda x: "blue_state"
        if x in blue_states
        else "red_state"
        if x in red_states
        else "swing_state"
        if x in swing_states
        else np.nan
    )

    masks: dict[tuple[str, str], pd.Series] = {("baseline", "baseline"): pd.Series(True, index=df.index)}
    for group in ["democrat", "republican", "independent"]:
        masks[("explicit_political", group)] = pid3.eq(group)
    for group in ["black_man", "black_woman", "white_man", "white_woman"]:
        masks[("explicit_demographic", group)] = race_gender.eq(group)
        masks[("implicit_demographic", group)] = race_gender.eq(group)
    for group in ["blue_state", "red_state", "swing_state"]:
        masks[("implicit_political", group)] = state_part.eq(group)
    return masks


def ces_rows(ces_df: pd.DataFrame, ground: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    masks = subgroup_masks(ces_df)
    weights = ces_df[weight_col]
    rows = []
    for _, issue in ground.iterrows():
        item = issue["ces_variable"]
        if item not in ces_df.columns:
            continue
        libv = libify(ces_df[item], int(issue["liberal_sign"]), item)
        population = weighted_mean(libv, weights)
        for family, group, label in CUE_ORDER:
            mask = masks[(family, group)]
            subgroup = weighted_mean(libv[mask], weights[mask])
            rows.append(
                {
                    "cue_family": family,
                    "cue_group": group,
                    "cue_label": label,
                    "ces_variable": item,
                    "ces_population_score": population,
                    "ces_score": subgroup,
                    "ces_shift": subgroup - population,
                }
            )
    return pd.DataFrame(rows)


def load_model_rows(evaluated_path: str | Path, ground: pd.DataFrame) -> pd.DataFrame:
    usecols = [
        "ces_variable",
        "cue_family",
        "cue_group",
        "cue_condition",
        "liberal_score",
        "baseline_liberal_score",
        "cue_effect",
    ]
    df = pd.read_csv(evaluated_path, usecols=usecols)
    df = df[df["ces_variable"].isin(set(ground["ces_variable"]))].copy()
    for col in ["liberal_score", "baseline_liberal_score", "cue_effect"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_shift"] = df["cue_effect"]
    missing = df["model_shift"].isna()
    df.loc[missing, "model_shift"] = (
        df.loc[missing, "liberal_score"] - df.loc[missing, "baseline_liberal_score"]
    )
    df.loc[df["cue_condition"].eq("baseline"), "model_shift"] = 0.0
    return df


def summarize_row_bootstrap(
    df: pd.DataFrame,
    value_col: str,
    rng: np.random.Generator,
    n_boot: int,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    for family, group, label in CUE_ORDER:
        arr = df.loc[df["cue_family"].eq(family) & df["cue_group"].eq(group), value_col]
        arr = arr.dropna().to_numpy(dtype="float64")
        if len(arr) == 0:
            rows.append(
                {
                    "cue_family": family,
                    "cue_group": group,
                    "cue_label": label,
                    f"{prefix}_mean": np.nan,
                    f"{prefix}_ci_low": np.nan,
                    f"{prefix}_ci_high": np.nan,
                    f"{prefix}_n": 0,
                }
            )
            continue
        boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
        lo, hi = ci(boot)
        rows.append(
            {
                "cue_family": family,
                "cue_group": group,
                "cue_label": label,
                f"{prefix}_mean": float(arr.mean()),
                f"{prefix}_ci_low": lo,
                f"{prefix}_ci_high": hi,
                f"{prefix}_n": len(arr),
            }
        )
    return pd.DataFrame(rows)


def summarize_ces_bootstrap(
    ces_df: pd.DataFrame,
    ground: pd.DataFrame,
    weight_col: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows = ces_rows(ces_df, ground, weight_col)
    point = (
        point_rows.groupby(["cue_family", "cue_group", "cue_label"], as_index=False)
        .agg(
            ces_score_mean=("ces_score", "mean"),
            ces_shift_mean=("ces_shift", "mean"),
            ces_population_mean=("ces_population_score", "mean"),
            ces_n=("ces_shift", "count"),
        )
    )

    n = len(ces_df)
    score_boot = {(family, group): [] for family, group, _label in CUE_ORDER}
    shift_boot = {(family, group): [] for family, group, _label in CUE_ORDER}
    pop_boot = []
    for _ in range(n_boot):
        sampled = ces_df.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        rows = ces_rows(sampled, ground, weight_col)
        grouped = rows.groupby(["cue_family", "cue_group"])[["ces_score", "ces_shift"]].mean()
        pop_boot.append(float(rows.drop_duplicates("ces_variable")["ces_population_score"].mean()))
        for key in score_boot:
            if key in grouped.index:
                score_boot[key].append(float(grouped.loc[key, "ces_score"]))
                shift_boot[key].append(float(grouped.loc[key, "ces_shift"]))

    ci_rows = []
    for family, group, label in CUE_ORDER:
        score_lo, score_hi = ci(np.array(score_boot[(family, group)], dtype="float64"))
        shift_lo, shift_hi = ci(np.array(shift_boot[(family, group)], dtype="float64"))
        ci_rows.append(
            {
                "cue_family": family,
                "cue_group": group,
                "cue_label": label,
                "ces_score_ci_low": score_lo,
                "ces_score_ci_high": score_hi,
                "ces_shift_ci_low": shift_lo,
                "ces_shift_ci_high": shift_hi,
            }
        )

    summary = point.merge(pd.DataFrame(ci_rows), on=["cue_family", "cue_group", "cue_label"], how="right")
    pop_lo, pop_hi = ci(np.array(pop_boot, dtype="float64"))
    summary["ces_population_ci_low"] = pop_lo
    summary["ces_population_ci_high"] = pop_hi
    return summary, point_rows


def order_estimates(df: pd.DataFrame) -> pd.DataFrame:
    order = {(family, group): idx for idx, (family, group, _label) in enumerate(CUE_ORDER)}
    out = df.copy()
    out["_order"] = [order[(family, group)] for family, group in zip(out["cue_family"], out["cue_group"])]
    return out.sort_values("_order").drop(columns="_order")


def plot_levels(estimates: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    x = np.arange(len(estimates))
    width = 0.34
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    ces_y = estimates["ces_score_mean"].to_numpy(dtype="float64")
    model_y = estimates["model_score_mean"].to_numpy(dtype="float64")
    ces_err = np.vstack(
        [ces_y - estimates["ces_score_ci_low"], estimates["ces_score_ci_high"] - ces_y]
    )
    model_err = np.vstack(
        [model_y - estimates["model_score_ci_low"], estimates["model_score_ci_high"] - model_y]
    )
    colors = [GROUP_COLORS[family] for family in estimates["cue_family"]]

    ax.bar(
        x - width / 2,
        ces_y,
        width=width,
        color="white",
        edgecolor="#2C4055",
        linewidth=1.3,
        yerr=ces_err,
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#2C4055"},
        label="CES weighted subgroup mean",
    )
    ax.bar(
        x + width / 2,
        model_y,
        width=width,
        color=colors,
        edgecolor="#222222",
        linewidth=0.7,
        yerr=model_err,
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#111111"},
        label="Model mean",
    )

    baseline_model = float(estimates.loc[estimates["cue_family"].eq("baseline"), "model_score_mean"].iloc[0])
    population_ces = float(estimates.loc[estimates["cue_family"].eq("baseline"), "ces_score_mean"].iloc[0])
    ax.axhline(baseline_model, color="#8C9A9A", linestyle="--", linewidth=1.3)
    ax.axhline(population_ces, color="#2C4055", linestyle=":", linewidth=1.3)

    ax.set_title(
        "Model output vs CES subgroup opinion, by cue group\n"
        "95% bootstrap CIs.  Dashed line: model no-cue baseline.  Dotted line: CES population mean.",
        loc="left",
        fontsize=18,
        pad=12,
    )
    ax.set_ylabel("Mean liberal-score (-1 conservative ... +1 liberal)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(estimates["cue_label"], rotation=42, ha="right", fontsize=10)
    ax.set_ylim(-0.65, 0.95)
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_effects(estimates: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    effects = estimates[~estimates["cue_family"].eq("baseline")].copy()
    effects = effects.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(effects))
    fig, ax = plt.subplots(figsize=(12.5, 9.2))

    band_colors = {
        "explicit_political": "#F4F6FB",
        "explicit_demographic": "#FCF5F3",
        "implicit_political": "#F4FAFD",
        "implicit_demographic": "#FCF6EF",
    }
    for family in effects["cue_family"].unique():
        idx = np.where(effects["cue_family"].to_numpy() == family)[0]
        ax.axhspan(idx.min() - 0.5, idx.max() + 0.5, color=band_colors[family], zorder=0)
        ax.text(0.60, idx.mean(), GROUP_LABELS[family], color=GROUP_COLORS[family], fontsize=13, weight="bold")

    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.7, alpha=0.8)
    colors = [GROUP_COLORS[family] for family in effects["cue_family"]]
    x = effects["model_shift_mean"].to_numpy(dtype="float64")
    xerr = np.vstack([x - effects["model_shift_ci_low"], effects["model_shift_ci_high"] - x])
    for idx, color in enumerate(colors):
        ax.errorbar(
            x[idx],
            y[idx],
            xerr=np.array([[xerr[0, idx]], [xerr[1, idx]]]),
            fmt="none",
            ecolor=color,
            capsize=4,
            linewidth=1.4,
            zorder=2,
        )
    ax.scatter(x, y, s=70, color=colors, zorder=3)

    labels = [label.replace("\n", " ") for label in effects["cue_label"]]
    labels = [f'"{label}."' if label.startswith("I am") else label for label in labels]
    labels = [
        label.replace("Name: white male", "White-male first name")
        .replace("Name: white female", "White-female first name")
        .replace("Name: black male", "Black-male first name")
        .replace("Name: black female", "Black-female first name")
        for label in labels
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(-1.0, 0.95)
    ax.set_xlabel(
        "Cue effect on liberal-score (vs no-cue baseline)\n"
        "Negative = output shifts conservative   |   Positive = output shifts liberal",
        fontsize=13,
    )
    total_n = int(effects["model_shift_n"].sum())
    ax.set_title(
        "Cue effect on political stance in writing assistance\n"
        f"Mixed-effects-style group means, 95% bootstrap CIs, n={total_n:,} scored generations",
        loc="left",
        fontsize=18,
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_did(estimates: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    x = np.arange(len(estimates))
    width = 0.34
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    ces_y = estimates["ces_shift_mean"].to_numpy(dtype="float64")
    model_y = estimates["model_shift_mean"].to_numpy(dtype="float64")
    ces_err = np.vstack(
        [ces_y - estimates["ces_shift_ci_low"], estimates["ces_shift_ci_high"] - ces_y]
    )
    model_err = np.vstack(
        [model_y - estimates["model_shift_ci_low"], estimates["model_shift_ci_high"] - model_y]
    )
    colors = [GROUP_COLORS[family] for family in estimates["cue_family"]]
    ax.bar(
        x - width / 2,
        ces_y,
        width=width,
        color="white",
        edgecolor="#2C4055",
        linewidth=1.3,
        yerr=ces_err,
        error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#2C4055"},
    )
    ax.bar(
        x + width / 2,
        model_y,
        width=width,
        color=colors,
        edgecolor="#222222",
        linewidth=0.7,
        yerr=model_err,
        error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#111111"},
    )
    ax.set_title(
        "Cue effects as difference-in-differences\n"
        "95% bootstrap CIs.  Model: cued - baseline.  CES: subgroup - population.",
        loc="left",
        fontsize=18,
        pad=12,
    )
    ax.set_ylabel("Mean liberal-score shift", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(estimates["cue_label"], rotation=42, ha="right", fontsize=10)
    ax.set_ylim(-0.95, 0.95)
    ax.tick_params(axis="y", labelsize=11)
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="#2C4055", label="CES shift"),
        mpatches.Patch(facecolor=GROUP_COLORS["baseline"], edgecolor="#222222", label="Model no-cue baseline"),
        mpatches.Patch(facecolor=GROUP_COLORS["explicit_political"], label=GROUP_LABELS["explicit_political"]),
        mpatches.Patch(facecolor=GROUP_COLORS["explicit_demographic"], label=GROUP_LABELS["explicit_demographic"]),
        mpatches.Patch(facecolor=GROUP_COLORS["implicit_political"], label=GROUP_LABELS["implicit_political"]),
        mpatches.Patch(facecolor=GROUP_COLORS["implicit_demographic"], label=GROUP_LABELS["implicit_demographic"]),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncols=2, fontsize=10, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ground = pd.read_csv(args.ground)
    ground = ground[ground["ces_variable"].notna()].copy()
    ces_columns = ["race", "gender4", "pid3", "inputstate", args.weight_col] + ground["ces_variable"].tolist()
    ces_df, _meta = pyreadstat.read_dta(args.ces_dta, usecols=ces_columns)

    model_rows = load_model_rows(args.evaluated, ground)
    model_score = summarize_row_bootstrap(model_rows, "liberal_score", rng, args.bootstrap, "model_score")
    model_shift = summarize_row_bootstrap(model_rows.dropna(subset=["model_shift"]), "model_shift", rng, args.bootstrap, "model_shift")
    ces_summary, ces_issue_rows = summarize_ces_bootstrap(ces_df, ground, args.weight_col, rng, args.bootstrap)

    estimates = ces_summary.merge(model_score, on=["cue_family", "cue_group", "cue_label"], how="left")
    estimates = estimates.merge(model_shift, on=["cue_family", "cue_group", "cue_label"], how="left")
    estimates["model_minus_ces_score"] = estimates["model_score_mean"] - estimates["ces_score_mean"]
    estimates["model_minus_ces_shift"] = estimates["model_shift_mean"] - estimates["ces_shift_mean"]
    estimates = order_estimates(estimates)

    estimates.to_csv(results_dir / "cue_ces_estimates.csv", index=False)
    ces_issue_rows.to_csv(results_dir / "cue_ces_by_issue.csv", index=False)

    plot_levels(estimates, figures_dir / "model_vs_ces_levels.png")
    plot_effects(estimates, figures_dir / "model_cue_effects.png")
    plot_did(estimates, figures_dir / "model_vs_ces_did.png")

    print(f"Wrote {figures_dir / 'model_vs_ces_levels.png'}")
    print(f"Wrote {figures_dir / 'model_cue_effects.png'}")
    print(f"Wrote {figures_dir / 'model_vs_ces_did.png'}")
    print(f"Wrote {results_dir / 'cue_ces_estimates.csv'}")
    print(f"Wrote {results_dir / 'cue_ces_by_issue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
