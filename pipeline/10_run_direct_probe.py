#!/usr/bin/env python3
"""Run the neutral-framed direct-inference probe (does the model refuse on clear cues?).

For each name x attribute (gender/race/political), ask a plain question and record
whether the model commits to an answer, commits with a caveat, or refuses/hedges.
Reuses the local-HF generation runtime; runs on the same open models as the rest.

    python pipeline/10_run_direct_probe.py --model <path> --device cuda:0 \
        --out-csv results/full/direct_probe_<tag>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_GEN_MODEL, DEFAULT_RESULTS_DIR
from direct_probe import DIRECT_ATTRIBUTES, DIRECT_NAMES, build_direct_prompt, parse_direct
from io_utils import append_jsonl, existing_prompt_ids, read_jsonl, write_csv
from probe_runtime import generate_batch_with_fallback, load_model
from prompting import slugify, stable_seed

DIRECT_COLUMNS = [
    "prompt_id", "name", "true_gender", "true_race", "attribute", "repeat", "seed",
    "prompt_text", "probe_model", "response_text", "label", "finish_reason",
]


def build_rows(repeats: int) -> list[dict]:
    rows = []
    for name, gender, race in DIRECT_NAMES:
        for attr in DIRECT_ATTRIBUTES:
            for rep in range(1, repeats + 1):
                pid = f"{slugify(name)}__{attr}__r{rep:02d}"
                rows.append({
                    "prompt_id": pid, "name": name, "true_gender": gender,
                    "true_race": race or "", "attribute": attr, "repeat": str(rep),
                    "seed": str(stable_seed(pid)), "prompt_text": build_direct_prompt(name, attr),
                })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_GEN_MODEL)
    p.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "direct_probe.jsonl"))
    p.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "direct_probe.csv"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-input-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.overwrite and Path(args.out_jsonl).exists():
        Path(args.out_jsonl).unlink()
    rows = build_rows(args.repeats)
    done = existing_prompt_ids(args.out_jsonl)
    pending = [r for r in rows if r["prompt_id"] not in done]
    print(f"{len(rows)} direct-probe rows; {len(pending)} pending.")
    if not pending:
        write_csv(args.out_csv, read_jsonl(args.out_jsonl), DIRECT_COLUMNS)
        return 0

    tokenizer, model, torch, input_device = load_model(args.model, args.device)
    args.device = input_device
    written = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        outs = generate_batch_with_fallback(batch, tokenizer, model, torch, args)
        for row, (resp, finish) in zip(batch, outs):
            out = dict(row)
            out.update(probe_model=args.model, response_text=resp,
                       label=parse_direct(row["attribute"], resp), finish_reason=finish)
            append_jsonl(args.out_jsonl, out)
            written += 1
    all_rows = read_jsonl(args.out_jsonl)
    write_csv(args.out_csv, all_rows, DIRECT_COLUMNS)
    print(f"Saved {len(all_rows)} direct probes -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
