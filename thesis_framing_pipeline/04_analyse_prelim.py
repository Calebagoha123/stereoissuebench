#!/usr/bin/env python3
"""Create preliminary CSV summaries and figures from evaluated rows."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from config import DEFAULT_RESULTS_DIR
from io_utils import read_table, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluated", default=str(DEFAULT_RESULTS_DIR / "evaluated_pilot.csv"))
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR / "analysis_pilot"))
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def numeric(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def se(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1)) / math.sqrt(len(values))


def group_summary(rows: list[dict], group_cols: list[str], value_col: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(col, "") for col in group_cols)].append(row)
    out = []
    for key, group_rows in sorted(grouped.items()):
        values = [numeric(row.get(value_col, "")) for row in group_rows]
        values = [value for value in values if value is not None]
        parse_errors = sum(1 for row in group_rows if row.get("eval_label") == "PARSE_ERROR")
        refusals = sum(1 for row in group_rows if row.get("eval_label") == "refusal")
        item = {col: value for col, value in zip(group_cols, key)}
        item.update(
            {
                "n": str(len(group_rows)),
                "n_scored": str(len(values)),
                "mean": f"{mean(values):.6f}" if values else "",
                "se": f"{se(values):.6f}" if len(values) > 1 else "",
                "parse_error_rate": f"{parse_errors / len(group_rows):.6f}",
                "refusal_rate": f"{refusals / len(group_rows):.6f}",
            }
        )
        out.append(item)
    return out


def add_baseline_effects(rows: list[dict]) -> list[dict]:
    baseline: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("cue_condition") == "baseline":
            score = numeric(row.get("liberal_score", ""))
            if score is not None:
                baseline[
                    (row["issue_id"], row["template_id"], str(row["generation_repeat"]))
                ] = score

    out = []
    missing = 0
    for row in rows:
        score = numeric(row.get("liberal_score", ""))
        key = (row["issue_id"], row["template_id"], str(row["generation_repeat"]))
        base = baseline.get(key)
        new_row = dict(row)
        new_row["baseline_liberal_score"] = "" if base is None else f"{base:.0f}"
        if row.get("cue_condition") == "baseline":
            new_row["cue_effect"] = "0"
        elif score is not None and base is not None:
            new_row["cue_effect"] = f"{score - base:.0f}"
        else:
            new_row["cue_effect"] = ""
            if row.get("cue_condition") != "baseline":
                missing += 1
        out.append(new_row)
    if missing:
        print(f"Warning: {missing} non-baseline rows lack a scored matched baseline.")
    return out


def paired_delta(rows: list[dict], left_condition: str, right_condition: str, label: str) -> list[dict]:
    by_key: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        score = numeric(row.get("liberal_score", ""))
        if score is None:
            continue
        key = (
            row["issue_id"],
            row["template_id"],
            str(row["generation_repeat"]),
            row.get("issue_cluster", ""),
        )
        by_key[key][row["cue_condition"]] = score

    deltas = []
    for key, values in by_key.items():
        if left_condition in values and right_condition in values:
            deltas.append(
                {
                    "comparison": label,
                    "issue_id": key[0],
                    "template_id": key[1],
                    "generation_repeat": key[2],
                    "issue_cluster": key[3],
                    "left_condition": left_condition,
                    "right_condition": right_condition,
                    "left_minus_right_liberal_score": f"{values[left_condition] - values[right_condition]:.0f}",
                }
            )
    return deltas


def category_means(rows: list[dict], family: str, group_col: str = "cue_group") -> list[dict]:
    subset = [row for row in rows if row.get("cue_family") == family]
    return group_summary(subset, [group_col], "liberal_score")


def paired_group_delta(
    rows: list[dict],
    family: str,
    left_group: str,
    right_group: str,
    label: str,
) -> list[dict]:
    by_key: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("cue_family") != family:
            continue
        score = numeric(row.get("liberal_score", ""))
        if score is None:
            continue
        key = (
            row["issue_id"],
            row["template_id"],
            str(row["generation_repeat"]),
            row.get("issue_cluster", ""),
        )
        by_key[key][row.get("cue_group", "")].append(score)

    out = []
    for key, groups in by_key.items():
        if groups.get(left_group) and groups.get(right_group):
            left = mean(groups[left_group])
            right = mean(groups[right_group])
            out.append(
                {
                    "comparison": label,
                    "issue_id": key[0],
                    "template_id": key[1],
                    "generation_repeat": key[2],
                    "issue_cluster": key[3],
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_mean_liberal_score": f"{left:.6f}",
                    "right_mean_liberal_score": f"{right:.6f}",
                    "left_minus_right_liberal_score": f"{left - right:.6f}",
                }
            )
    return out


def demographic_construct_comparison(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("cue_family") not in {"explicit_demographic", "implicit_demographic"}:
            continue
        score = numeric(row.get("liberal_score", ""))
        if score is None:
            continue
        key = (
            row["issue_id"],
            row["template_id"],
            str(row["generation_repeat"]),
            row.get("issue_cluster", ""),
            row.get("cue_group", ""),
        )
        by_key[key][row["cue_family"]].append(score)

    out = []
    for key, families in by_key.items():
        if families.get("explicit_demographic") and families.get("implicit_demographic"):
            explicit = mean(families["explicit_demographic"])
            implicit = mean(families["implicit_demographic"])
            out.append(
                {
                    "comparison": "explicit_demographic_minus_name_cue",
                    "issue_id": key[0],
                    "template_id": key[1],
                    "generation_repeat": key[2],
                    "issue_cluster": key[3],
                    "demographic_group": key[4],
                    "explicit_mean_liberal_score": f"{explicit:.6f}",
                    "name_cue_mean_liberal_score": f"{implicit:.6f}",
                    "explicit_minus_name_cue_liberal_score": f"{explicit - implicit:.6f}",
                }
            )
    return out


def make_figures(out_dir: Path, condition_summary: list[dict], issue_summary: list[dict], diagnostics: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV summaries were written, figures skipped.")
        return

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    non_base = [row for row in condition_summary if row.get("cue_condition") != "baseline" and row.get("mean")]
    non_base.sort(key=lambda row: float(row["mean"]))
    fig, ax = plt.subplots(figsize=(9, max(5, len(non_base) * 0.28)))
    ax.scatter([float(row["mean"]) for row in non_base], range(len(non_base)), color="#2f6f9f")
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(non_base)))
    ax.set_yticklabels([row["cue_condition"] for row in non_base], fontsize=7)
    ax.set_xlabel("Mean liberal-score cue effect vs matched baseline")
    fig.tight_layout()
    fig.savefig(figures_dir / "cue_effect_dot_plot.png", dpi=180)
    plt.close(fig)

    cues = sorted({row["cue_condition"] for row in issue_summary if row.get("cue_condition") != "baseline"})
    issues = sorted({row["issue_cluster"] for row in issue_summary})
    matrix = []
    for issue in issues:
        row_vals = []
        for cue in cues:
            match = next(
                (
                    row
                    for row in issue_summary
                    if row["issue_cluster"] == issue and row["cue_condition"] == cue and row.get("mean")
                ),
                None,
            )
            row_vals.append(float(match["mean"]) if match else 0.0)
        matrix.append(row_vals)
    fig, ax = plt.subplots(figsize=(max(10, len(cues) * 0.35), max(4, len(issues) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cues)))
    ax.set_xticklabels(cues, rotation=90, fontsize=6)
    ax.set_yticks(range(len(issues)))
    ax.set_yticklabels(issues, fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean liberal-score cue effect")
    fig.tight_layout()
    fig.savefig(figures_dir / "issue_by_cue_heatmap.png", dpi=180)
    plt.close(fig)

    party_rows = [
        row
        for row in condition_summary
        if row.get("cue_condition")
        in {"explicit_political_democrat", "explicit_political_republican", "explicit_political_independent"}
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [row["cue_condition"].replace("explicit_political_", "") for row in party_rows],
        [float(row["mean"]) if row.get("mean") else 0 for row in party_rows],
        color=["#3f6fb5", "#b54848", "#777777"][: len(party_rows)],
    )
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_ylabel("Mean liberal-score cue effect")
    fig.tight_layout()
    fig.savefig(figures_dir / "party_control_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, max(4, len(diagnostics) * 0.22)))
    diagnostics = sorted(diagnostics, key=lambda row: row["cue_condition"])
    ax.barh(
        [row["cue_condition"] for row in diagnostics],
        [float(row["parse_error_rate"]) + float(row["refusal_rate"]) for row in diagnostics],
        color="#777777",
    )
    ax.set_xlabel("Parse error + refusal rate")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "parse_refusal_diagnostics.png", dpi=180)
    plt.close(fig)
    print(f"Figures written to {figures_dir}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_table(args.evaluated)
    rows = add_baseline_effects(rows)
    write_csv(out_dir / "evaluated_with_effects.csv", rows)

    condition_summary = group_summary(
        [row for row in rows if row.get("cue_condition") != "baseline"],
        ["cue_condition", "cue_family", "cue_group", "cue_value"],
        "cue_effect",
    )
    issue_summary = group_summary(
        [row for row in rows if row.get("cue_condition") != "baseline"],
        ["issue_cluster", "cue_condition"],
        "cue_effect",
    )
    diagnostics = group_summary(rows, ["cue_condition"], "liberal_score")

    write_csv(out_dir / "cue_effects_by_condition.csv", condition_summary)
    write_csv(out_dir / "cue_effects_by_issue_condition.csv", issue_summary)
    write_csv(out_dir / "parse_refusal_diagnostics.csv", diagnostics)

    party_delta = paired_delta(
        rows,
        "explicit_political_democrat",
        "explicit_political_republican",
        "democrat_minus_republican",
    )
    write_csv(out_dir / "party_positive_control.csv", party_delta)

    write_csv(out_dir / "implicit_political_category_scores.csv", category_means(rows, "implicit_political"))
    write_csv(
        out_dir / "implicit_political_blue_vs_red.csv",
        paired_group_delta(
            rows,
            "implicit_political",
            "blue_state",
            "red_state",
            "blue_state_minus_red_state",
        ),
    )
    demographic_scores = category_means(rows, "explicit_demographic") + category_means(rows, "implicit_demographic")
    write_csv(out_dir / "demographic_construct_validity_scores.csv", demographic_scores)
    write_csv(
        out_dir / "demographic_construct_validity_comparison.csv",
        demographic_construct_comparison(rows),
    )

    if not args.skip_figures:
        make_figures(out_dir, condition_summary, issue_summary, diagnostics)

    parse_errors = sum(1 for row in rows if row.get("eval_label") == "PARSE_ERROR")
    print(f"Analysed {len(rows)} evaluated rows.")
    print(f"Parse errors: {parse_errors}/{len(rows)} ({parse_errors / len(rows):.2%})")
    print(f"CSV summaries written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
