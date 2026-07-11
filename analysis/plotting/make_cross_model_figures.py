#!/usr/bin/env python3
"""Cross-group x cross-model figures for the two-arm cue-steering run.

Reads the three judged eval files (one per generation model) produced by
``03_run_stance_eval.py`` and renders the headline comparison figures into
``--figures-dir``. Stance is the per-response liberal score in {-1, 0, +1}
(−1 = writes the conservative side of the issue, +1 = the liberal side); a
cue's effect is its mean liberal score minus the model's baseline mean.

Inputs (defaults): results/full/eval_{llama,gemma,qwen}.csv
Outputs: figures/full/{arm_a_cue_effects,arm_b_groups,demographic_gradient,refusals}.png
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

MODELS = ["llama", "gemma", "qwen"]
MODEL_LABEL = {"llama": "Llama-3.1-8B", "gemma": "Gemma-3-12B", "qwen": "Qwen3.6-27B"}
MODEL_COLOUR = {"llama": "#2e6da4", "gemma": "#27915b", "qwen": "#c0392b"}

ARM_A_CUES = [
    ("explicit_political_democrat", "Democrat"),
    ("explicit_political_republican", "Republican"),
    ("explicit_political_independent", "Independent"),
    ("explicit_demographic_black_woman", "Black woman"),
    ("explicit_demographic_black_man", "Black man"),
    ("explicit_demographic_white_woman", "White woman"),
    ("explicit_demographic_white_man", "White man"),
]
ARM_B_GROUPS = [
    ("black_woman", "Black woman"), ("black_man", "Black man"),
    ("white_woman", "White woman"), ("white_man", "White man"),
    ("blue_state", "Blue state"), ("swing_state", "Swing state"),
    ("red_state", "Red state"),
]


def load(results_dir: Path, model: str) -> list[dict]:
    import csv
    with (results_dir / f"eval_{model}.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _num(r: dict) -> float | None:
    v = r["liberal_score"]
    return float(v) if v not in ("", "None") else None


def mean_se_clustered(rows: list[dict], cluster_key: str | None) -> tuple[float, float, int]:
    """Mean liberal score; SE clustered on ``cluster_key`` (e.g. instance_id) if given."""
    vals = [_num(r) for r in rows if _num(r) is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / len(vals)
    if cluster_key is None:
        # pooled SE
        if len(vals) < 2:
            return mean, float("nan"), len(vals)
        sd = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        return mean, sd / math.sqrt(len(vals)), len(vals)
    by = defaultdict(list)
    for r in rows:
        v = _num(r)
        if v is not None:
            by[r[cluster_key]].append(v)
    cmeans = [sum(v) / len(v) for v in by.values() if v]
    if len(cmeans) < 2:
        return mean, float("nan"), len(cmeans)
    cm = sum(cmeans) / len(cmeans)
    sd = (sum((x - cm) ** 2 for x in cmeans) / (len(cmeans) - 1)) ** 0.5
    return mean, sd / math.sqrt(len(cmeans)), len(cmeans)


def summarise(data: dict[str, list[dict]]):
    """Per model: baseline mean, Arm-A cue means/SE, Arm-B group means/SE, refusals."""
    out = {}
    for m, rows in data.items():
        armA = [r for r in rows if r["arm"] == "A"]
        armB = [r for r in rows if r["arm"] == "B"]
        base_rows = [r for r in armA if r["cue_condition"] == "baseline"]
        base = mean_se_clustered(base_rows, None)[0]
        cueA = {c: mean_se_clustered([r for r in armA if r["cue_condition"] == c], None)
                for c, _ in ARM_A_CUES}
        grpB = {g: mean_se_clustered([r for r in armB if r["cue_group"] == g], "instance_id")
                for g, _ in ARM_B_GROUPS}
        # refusal rate per Arm-A cue
        refA = {}
        for c, _ in [("baseline", "")] + ARM_A_CUES:
            cr = [r for r in armA if r["cue_condition"] == c]
            refA[c] = (sum(r["eval_label"] == "refusal" for r in cr) / len(cr)) if cr else 0.0
        out[m] = dict(base=base, cueA=cueA, grpB=grpB, refA=refA)
    return out


def _grouped_bars(ax, labels, series, errs=None):
    n = len(series)
    width = 0.8 / n
    x = range(len(labels))
    for i, m in enumerate(MODELS):
        offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
        e = errs[m] if errs else None
        ax.bar(offs, series[m], width=width, label=MODEL_LABEL[m],
               color=MODEL_COLOUR[m], yerr=e, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(0, color="#333", lw=0.8)


def fig_arm_a(summ, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [lbl for _, lbl in ARM_A_CUES]
    series = {m: [summ[m]["cueA"][c][0] - summ[m]["base"] for c, _ in ARM_A_CUES] for m in MODELS}
    errs = {m: [summ[m]["cueA"][c][1] for c, _ in ARM_A_CUES] for m in MODELS}
    _grouped_bars(ax, labels, series, errs)
    ax.set_ylabel("Δ mean liberal score vs baseline")
    ax.set_title("Arm A — explicit cue effect on stance, by model\n"
                 "(+ = pulls toward liberal side, − = toward conservative side)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "arm_a_cue_effects.png"); plt.close(fig)


def fig_arm_b(summ, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [lbl for _, lbl in ARM_B_GROUPS]
    series = {m: [summ[m]["grpB"][g][0] for g, _ in ARM_B_GROUPS] for m in MODELS}
    errs = {m: [summ[m]["grpB"][g][1] for g, _ in ARM_B_GROUPS] for m in MODELS}
    _grouped_bars(ax, labels, series, errs)
    for m in MODELS:
        ax.axhline(summ[m]["base"], color=MODEL_COLOUR[m], lw=0.9, ls="--", alpha=0.6)
    ax.set_ylabel("mean liberal score")
    ax.set_title("Arm B — sampled-instance cues (names & states rotated within group)\n"
                 "dashed line = each model's baseline; error bars clustered on instance")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "arm_b_groups.png"); plt.close(fig)


def fig_gradient(summ, out):
    """Explicit vs implicit demographic gradient, white_man..black_woman."""
    order = [("white_man", "White\nman"), ("white_woman", "White\nwoman"),
             ("black_man", "Black\nman"), ("black_woman", "Black\nwoman")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, arm, title in [(axes[0], "A", "Explicit label (“As a Black man”)"),
                           (axes[1], "B", "Implicit name (rotated, e.g. “Jamal”)")]:
        for m in MODELS:
            if arm == "A":
                ys = [summ[m]["cueA"][f"explicit_demographic_{g}"][0] - summ[m]["base"] for g, _ in order]
            else:
                ys = [summ[m]["grpB"][g][0] - summ[m]["base"] for g, _ in order]
            ax.plot(range(len(order)), ys, "-o", color=MODEL_COLOUR[m], label=MODEL_LABEL[m])
        ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order])
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Δ mean liberal score vs baseline")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Demographic stance gradient: explicit vs implicit (name) cue", y=1.02)
    fig.tight_layout(); fig.savefig(out / "demographic_gradient.png", bbox_inches="tight"); plt.close(fig)


CUE_FAMILY_COLOUR = {
    "political": "#c0392b",      # explicit political
    "demographic": "#2e6da4",    # explicit demographic label (Arm A)
    "name": "#27915b",           # rotated name group (Arm B)
    "state": "#e08aa8",          # rotated state group (Arm B)
}


def fig_per_model(summ, model, out):
    """One self-contained figure per model: every cue's Δ vs baseline, both arms,
    coloured by cue family so the explicit/implicit and political/demographic
    structure is visible within the single model."""
    items = []  # (label, delta, se, family)
    for c, lbl in ARM_A_CUES:
        fam = "political" if "political" in c else "demographic"
        mn, se, _ = summ[model]["cueA"][c]
        items.append((f"{lbl}  (label)", mn - summ[model]["base"], se, fam))
    for g, lbl in ARM_B_GROUPS:
        fam = "state" if g.endswith("_state") else "name"
        mn, se, _ = summ[model]["grpB"][g]
        suffix = "(state)" if fam == "state" else "(name)"
        items.append((f"{lbl}  {suffix}", mn - summ[model]["base"], se, fam))

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ys = list(range(len(items)))[::-1]
    for y, (lbl, d, se, fam) in zip(ys, items):
        ax.barh(y, d, color=CUE_FAMILY_COLOUR[fam],
                xerr=se, capsize=2, error_kw={"lw": 0.8})
    ax.set_yticks(list(range(len(items)))[::-1])
    ax.set_yticklabels([lbl for lbl, *_ in items])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Δ mean liberal score vs baseline  (+ liberal / − conservative)")
    ax.set_title(f"{MODEL_LABEL[model]} — cue effects on stance\n"
                 f"baseline mean liberal score = {summ[model]['base']:+.2f}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CUE_FAMILY_COLOUR[f])
               for f in ["political", "demographic", "name", "state"]]
    ax.legend(handles, ["Explicit political", "Explicit demographic (label)",
                        "Implicit demographic (name)", "Implicit political (state)"],
              frameon=False, fontsize=8, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(out / f"cue_effects_{model}.png", bbox_inches="tight")
    plt.close(fig)


def fig_refusals(summ, out):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cues = [("baseline", "baseline")] + ARM_A_CUES
    labels = [lbl for _, lbl in cues]
    series = {m: [100 * summ[m]["refA"][c] for c, _ in cues] for m in MODELS}
    _grouped_bars(ax, labels, series)
    ax.set_ylabel("refusal rate (%)")
    ax.set_title("Arm A refusal rate by cue, per model")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "refusals.png"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full")
    ap.add_argument("--figures-dir", default="figures/full")
    args = ap.parse_args()
    rd, fd = Path(args.results_dir), Path(args.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    data = {m: load(rd, m) for m in MODELS}
    summ = summarise(data)
    fig_arm_a(summ, fd)
    fig_arm_b(summ, fd)
    fig_gradient(summ, fd)
    fig_refusals(summ, fd)
    for m in MODELS:
        fig_per_model(summ, m, fd)
    print(f"Wrote {4 + len(MODELS)} figures to {fd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
