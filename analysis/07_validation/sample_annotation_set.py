"""Draw the human-annotation validation sample for the DeBERTa stance classifier.

Design (see thesis Appendix A):
  - Frame: full_3x bert_eval score files, per model.
  - Target 250 total across the FULL 5-model set = 50/model. The two frontier models
    aren't generated yet, so this draws 50 x (models present) now (150 for the 3
    open-weight models) and the remaining 100 is drawn later with --start-index 151
    once GPT-5.4 / Claude Sonnet 5 responses exist.
  - Within each model, 50 is split across 3 score-bins (~equal thirds):
      * Equal thirds by score oversamples the scarce conservative tail (0-40, ~11%
        of the corpus) and puts mass either side of the 40 / 60 decision thresholds,
        so the same sample doubles as the input to the threshold-sensitivity check.
      * 50/model supports the per-model robustness table (a check, not a claim: cue
        effects are within-model diffs, so a constant per-model offset cancels).
  - prompt_id is unique ACROSS the whole sample: the same issue x template x cue x
    repeat is never drawn for two different models (prompt_id is shared across models,
    so cells are sampled with a global exclusion set).
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


def load_rows(bert_dir: Path, models) -> list[dict]:
    rows: list[dict] = []
    for model in models:
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
    ap.add_argument("--per-model", type=int, default=50,
                    help="items per model (50 x 5 models = 250 total)")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated models to draw now")
    ap.add_argument("--start-index", type=int, default=1,
                    help="item_id numbering start (use 151 when adding the 2 frontier models)")
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = load_rows(args.bert_dir, models)
    pop = Counter((r["model"], r["bin"]) for r in rows)
    bin_names = [b[0] for b in BINS]

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["model"], r["bin"])].append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    used: set[str] = set()  # prompt_ids already taken (unique across the whole sample)
    for model in models:
        balloc = allocate(args.per_model, bin_names)
        for bname in bin_names:
            pool = [r for r in by_cell.get((model, bname), []) if r["prompt_id"] not in used]
            want = balloc[bname]
            if len(pool) < want:
                print(f"  WARN cell {(model, bname)}: only {len(pool)} available, wanted {want}")
                want = len(pool)
            chosen = rng.sample(pool, want)
            weight = pop[(model, bname)] / want if want else 0.0
            for r in chosen:
                r["pop_weight"] = round(weight, 4)
                used.add(r["prompt_id"])
                picked.append(r)

    rng.shuffle(picked)  # break up model/bin ordering for the annotator
    for i, r in enumerate(picked, args.start_index):
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
    for model in models:
        for bname in bin_names:
            cell = (model, bname)
            s = sum(1 for r in picked if (r["model"], r["bin"]) == cell)
            print(f"  {model:6s} {bname:3s}: {s:3d} / {pop[cell]}")
    print("By model:", dict(Counter(r["model"] for r in picked)))
    print("Unique prompt_ids:", len({r["prompt_id"] for r in picked}), "/", len(picked))
    print("By cue_family:", dict(Counter(r["cue_family"] for r in picked)))


if __name__ == "__main__":
    main()
