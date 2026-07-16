"""Draw the human-annotation validation sample for the DeBERTa stance classifier.

Design (see thesis Appendix A):
  - Frame: full_3x bert_eval score files for the 3 open-weight models.
  - n = 250, stratified as 3 models x 3 score-bins = 9 cells, ~equal allocation.
      * Equal thirds by score oversamples the scarce conservative tail (0-40, ~11%
        of the corpus) and puts mass either side of the 40 / 60 decision thresholds,
        so the same sample doubles as the input to the threshold-sensitivity check.
      * Equal by model gives >=~35/model for the per-model robustness table (a check,
        not a claim: cue effects are within-model diffs, so a constant per-model
        offset cancels).
  - Cue family is NOT stratified on; it is recorded so the cue-correlated-error check
    can be run post-hoc (that check is what actually guards the RQ1/RQ2 claims).
  - Each item carries a population weight = pop_count(cell) / sampled_count(cell) so
    agreement can be reweighted back to the deployment distribution.

Outputs analysis/07_validation/out/sample_keys.csv (blinding key: prompt_id, model,
score, cue, bin, weight). The response text is joined on Brains by build_items.py.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import random

MODELS = ("qwen", "gemma", "llama")
# The continuous 0-100 stance score the 40/60 thresholds apply to (0 = opposes the
# issue proposition, 100 = supports it). bert_liberal_score / bert_support_score are
# already collapsed to {-1,0,1}, so they are NOT the right column to bin on.
SCORE_COL = "bert_pred_stance"
# Score-bin edges matching the classifier's 40 / 60 decision thresholds.
BINS = (("con", 0.0, 40.0), ("neu", 40.0, 60.0), ("lib", 60.0, 100.01))


def bin_of(score: float) -> str:
    for name, lo, hi in BINS:
        if lo <= score < hi:
            return name
    return "lib"  # score == 100.0 guard


def load_rows(bert_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for model in MODELS:
        path = bert_dir / f"bert_eval_{model}.csv"
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                score = r.get(SCORE_COL, "")
                if score in ("", None):
                    continue
                rows.append(
                    {
                        "prompt_id": r["prompt_id"],
                        "model": model,
                        "score": float(score),
                        "bin": bin_of(float(score)),
                        "cue_family": r.get("cue_family", ""),
                        "cue_group": r.get("cue_group", ""),
                        "issue_id": r.get("issue_id", ""),
                        "stance_target": r.get("stance_target", ""),
                        "liberal_sign": r.get("liberal_sign", ""),
                        "bert_pred_stance": r.get("bert_pred_stance", ""),
                        "bert_collapsed_stance": r.get("bert_collapsed_stance", ""),
                    }
                )
    return rows


def allocate(n: int, cells: list[tuple]) -> dict[tuple, int]:
    """Spread n as evenly as possible across the given cells (largest-remainder)."""
    k = len(cells)
    base, rem = divmod(n, k)
    alloc = {c: base for c in cells}
    for c in cells[:rem]:  # deterministic: give the remainder to the first cells
        alloc[c] += 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-dir", default="results/full_3x", type=Path)
    ap.add_argument("--out", default="analysis/07_validation/out/sample_keys.csv", type=Path)
    ap.add_argument("-n", "--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    rows = load_rows(args.bert_dir)
    pop = Counter((r["model"], r["bin"]) for r in rows)
    cells = [(m, b[0]) for m in MODELS for b in BINS]
    alloc = allocate(args.n, cells)

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["model"], r["bin"])].append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for cell in cells:
        pool = by_cell.get(cell, [])
        want = alloc[cell]
        if len(pool) < want:
            print(f"  WARN cell {cell}: only {len(pool)} available, wanted {want}")
            want = len(pool)
        chosen = rng.sample(pool, want)
        weight = pop[cell] / want if want else 0.0
        for r in chosen:
            r["pop_weight"] = round(weight, 4)
            picked.append(r)

    rng.shuffle(picked)  # break up model/bin ordering for the annotator
    for i, r in enumerate(picked, 1):
        r["item_id"] = f"A{i:03d}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "item_id", "prompt_id", "model", "bin", "score", "pop_weight",
        "cue_family", "cue_group", "issue_id", "stance_target", "liberal_sign",
        "bert_pred_stance", "bert_collapsed_stance",
    ]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in picked:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"Wrote {len(picked)} items -> {args.out}")
    print("Per-cell allocation (sampled / population):")
    for cell in cells:
        s = sum(1 for r in picked if (r["model"], r["bin"]) == cell)
        print(f"  {cell[0]:6s} {cell[1]:3s}: {s:3d} / {pop[cell]}")
    print("By model:", dict(Counter(r["model"] for r in picked)))
    print("By cue_family:", dict(Counter(r["cue_family"] for r in picked)))


if __name__ == "__main__":
    main()
