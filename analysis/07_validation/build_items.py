"""Join the sampled prompt_ids to their response text to produce the blinded
annotation items. Runs on Brains, where the full_3x generation outputs live.

The annotator sees ONLY: the issue proposition (stance_target) and the response
text. Model, cue, and every classifier score stay in sample_keys.csv (the key) and
never reach the UI, so annotation is blind to both the cue and the model's own score.

Usage (on Brains):
    python analysis/07_validation/build_items.py \
        --keys analysis/07_validation/out/sample_keys.csv \
        --responses 'results/full_3x/gen_*.jsonl' \
        --out annotation/items.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

# Column / field names that may carry the response text or the join id, in priority order.
TEXT_KEYS = ("response_text", "response", "generation", "output_text", "text")
ID_KEYS = ("prompt_id", "id", "custom_id")
MODELS = ("qwen", "gemma", "llama", "gpt56terra", "sonnet5")


def _first(d: dict, keys) -> str | None:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def model_from_path(path: str) -> str:
    """The gen dumps are per-model (gen_<model>_arm_*). prompt_id is NOT unique
    across models (model lives in a separate field), so we must key on the model
    too, and the filename is the reliable source of it."""
    base = os.path.basename(path).lower()
    for m in MODELS:
        if m in base:
            return m
    raise SystemExit(f"Cannot infer model from filename: {path}")


def build_index(patterns: list[str]) -> dict[tuple[str, str], str]:
    """(prompt_id, model) -> response_text, scanning csv/jsonl response dumps."""
    idx: dict[tuple[str, str], str] = {}
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit(f"No response files matched: {patterns}")
    for path in files:
        model = model_from_path(path)
        if path.endswith(".jsonl"):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    pid, txt = _first(d, ID_KEYS), _first(d, TEXT_KEYS)
                    if pid and txt:
                        idx[(pid, model)] = txt
        else:  # csv
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    pid, txt = _first(r, ID_KEYS), _first(r, TEXT_KEYS)
                    if pid and txt:
                        idx[(pid, model)] = txt
        print(f"  scanned {path} (model={model}): index now {len(idx)}")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="analysis/07_validation/out/sample_keys.csv", type=Path)
    ap.add_argument("--responses", nargs="+", required=True,
                    help="glob(s) for full_3x response dumps (csv or jsonl)")
    ap.add_argument("--out", default="annotation/items.csv", type=Path)
    args = ap.parse_args()

    idx = build_index(args.responses)

    keys = list(csv.DictReader(args.keys.open(newline="")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "stance_target", "response_text"])
        w.writeheader()
        for r in keys:
            txt = idx.get((r["prompt_id"], r["model"]))
            if txt is None:
                missing.append((r["prompt_id"], r["model"]))
                continue
            w.writerow({
                "item_id": r["item_id"],
                "stance_target": r["stance_target"],
                "response_text": txt,
            })
    print(f"Wrote {len(keys) - len(missing)}/{len(keys)} items -> {args.out}")
    if missing:
        print(f"MISSING response text for {len(missing)} prompt_ids, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
