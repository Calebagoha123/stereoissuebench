#!/usr/bin/env python3
"""Merge sharded generation outputs and restore prompt-file order."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_RESULTS_DIR, GENERATION_COLUMNS
from io_utils import read_csv, read_table, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(DEFAULT_RESULTS_DIR / "prompts_pilot.csv"))
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Generation JSONL/CSV files to merge. Earlier files win duplicate prompt_ids.",
    )
    parser.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "generations_pilot_merged.jsonl"))
    parser.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "generations_pilot_merged.csv"))
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write available rows even if some prompt_ids are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_rows = read_csv(args.prompts)
    prompt_ids = [row["prompt_id"] for row in prompt_rows]
    prompt_id_set = set(prompt_ids)

    by_prompt_id: dict[str, dict] = {}
    duplicates = 0
    ignored = 0
    for path in args.inputs:
        path_obj = Path(path)
        if not path_obj.exists():
            raise SystemExit(f"Missing input file: {path}")
        for row in read_table(path_obj):
            prompt_id = row.get("prompt_id", "")
            if prompt_id not in prompt_id_set:
                ignored += 1
                continue
            if prompt_id in by_prompt_id:
                duplicates += 1
                continue
            by_prompt_id[prompt_id] = row

    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_prompt_id]
    if missing and not args.allow_missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"Missing {len(missing)} prompt_ids; first missing: {preview}. "
            "Use --allow-missing to write a partial merge."
        )

    merged = [by_prompt_id[prompt_id] for prompt_id in prompt_ids if prompt_id in by_prompt_id]
    write_jsonl(args.out_jsonl, merged)
    write_csv(args.out_csv, merged, GENERATION_COLUMNS)

    print(f"Merged rows: {len(merged)}/{len(prompt_ids)}")
    print(f"Duplicate rows skipped: {duplicates}")
    print(f"Rows ignored because prompt_id was not in prompts: {ignored}")
    print(f"Missing rows: {len(missing)}")
    print(f"Saved merged generations to {args.out_jsonl} and {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
