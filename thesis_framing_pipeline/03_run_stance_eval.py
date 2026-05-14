#!/usr/bin/env python3
"""Classify generated responses with a local Qwen stance judge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DEFAULT_JUDGE_MODEL, DEFAULT_RESULTS_DIR, EVAL_COLUMNS
from hf_utils import apply_chat_template, resolve_local_model_path
from io_utils import append_jsonl, existing_prompt_ids, read_jsonl, read_table, write_csv
from stance import build_eval_prompt, collapsed_stance, liberal_score, parse_label, support_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default=str(DEFAULT_RESULTS_DIR / "generations_pilot.csv"))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "evaluated_pilot.jsonl"))
    parser.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "evaluated_pilot.csv"))
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing JSONL/CSV outputs before running.",
    )
    parser.add_argument("--flush-every", type=int, default=100)
    return parser.parse_args()


def load_model(model_path: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing stance-eval dependencies. Install torch and transformers in the run environment."
        ) from exc

    resolved = resolve_local_model_path(model_path)
    print(f"Loading judge tokenizer from {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading judge model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, torch


def eval_batch(rows: list[dict], tokenizer, model, torch, args: argparse.Namespace) -> list[str]:
    prompts = [build_eval_prompt(row) for row in rows]
    formatted = [apply_chat_template(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(args.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
        for ids in output_ids
    ]


def eval_batch_with_fallback(
    rows: list[dict],
    tokenizer,
    model,
    torch,
    args: argparse.Namespace,
) -> list[str]:
    try:
        return eval_batch(rows, tokenizer, model, torch, args)
    except Exception as exc:
        if len(rows) == 1:
            return [f"EVAL_ERROR: {type(exc).__name__}: {exc}"]
        print(
            f"Batch eval failed for {len(rows)} rows; retrying row-by-row. "
            f"Error was {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        outputs = []
        for row in rows:
            outputs.extend(eval_batch_with_fallback([row], tokenizer, model, torch, args))
        return outputs


def main() -> int:
    args = parse_args()
    if args.overwrite:
        for path in [args.out_jsonl, args.out_csv]:
            path_obj = Path(path)
            if path_obj.exists():
                path_obj.unlink()
    generations = read_table(args.generations)
    if args.limit is not None:
        generations = generations[: args.limit]
    done = set() if args.no_resume else existing_prompt_ids(args.out_jsonl)
    pending = [row for row in generations if row["prompt_id"] not in done]
    print(f"Loaded {len(generations)} generations; {len(done)} already done; {len(pending)} pending.")
    if not pending:
        if Path(args.out_jsonl).exists():
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), EVAL_COLUMNS)
        return 0

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    tokenizer, model, torch = load_model(args.model, args.device)

    written = 0
    for start in range(0, len(pending), args.batch_size):
        batch_rows = pending[start : start + args.batch_size]
        raw_outputs = eval_batch_with_fallback(batch_rows, tokenizer, model, torch, args)
        for row, raw in zip(batch_rows, raw_outputs):
            label = parse_label(raw)
            s_score = support_score(label)
            l_score = liberal_score(label, row["liberal_sign"])
            out_row = dict(row)
            out_row["judge_model"] = args.model
            out_row["eval_text"] = raw
            out_row["eval_label"] = label
            out_row["collapsed_stance"] = collapsed_stance(label)
            out_row["support_score"] = "" if s_score is None else str(s_score)
            out_row["liberal_score"] = "" if l_score is None else str(l_score)
            append_jsonl(args.out_jsonl, out_row)
            written += 1
        if written % args.flush_every == 0:
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), EVAL_COLUMNS)
            print(f"Wrote {written}/{len(pending)} pending evaluations.")

    rows = read_jsonl(args.out_jsonl)
    write_csv(args.out_csv, rows, EVAL_COLUMNS)
    parse_errors = sum(1 for row in rows if row.get("eval_label") == "PARSE_ERROR")
    print(f"Saved evaluations to {args.out_jsonl} and {args.out_csv}")
    print(f"Parse errors: {parse_errors}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
