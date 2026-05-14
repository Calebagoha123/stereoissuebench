#!/usr/bin/env python3
"""Run local Qwen generation for prompt rows."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

from config import DEFAULT_GEN_MODEL, DEFAULT_RESULTS_DIR, GENERATION_COLUMNS
from hf_utils import apply_chat_template, resolve_local_model_path
from io_utils import append_jsonl, existing_prompt_ids, read_csv, read_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(DEFAULT_RESULTS_DIR / "prompts_pilot.csv"))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "generations_pilot.jsonl"))
    parser.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "generations_pilot.csv"))
    parser.add_argument("--model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--strict-row-seeds",
        action="store_true",
        help="Force batch size 1 so each row's stored seed is applied independently.",
    )
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
            "Missing generation dependencies. Install torch and transformers in the run environment."
        ) from exc

    resolved = resolve_local_model_path(model_path)
    print(f"Loading generation tokenizer from {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading generation model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, torch


def seed_for_batch(rows: list[dict], torch) -> None:
    seed_source = "|".join(f"{row['prompt_id']}:{row['seed']}" for row in rows)
    seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_batch(rows: list[dict], tokenizer, model, torch, args: argparse.Namespace) -> list[str]:
    seed_for_batch(rows, torch)
    formatted = [apply_chat_template(tokenizer, row["prompt_text"]) for row in rows]
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
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
        for ids in output_ids
    ]


def generate_batch_with_fallback(
    rows: list[dict],
    tokenizer,
    model,
    torch,
    args: argparse.Namespace,
) -> list[tuple[str, str]]:
    try:
        return [(response, "ok") for response in generate_batch(rows, tokenizer, model, torch, args)]
    except Exception as exc:
        if len(rows) == 1:
            return [(f"GENERATION_ERROR: {type(exc).__name__}: {exc}", "error")]
        print(
            f"Batch generation failed for {len(rows)} rows; retrying row-by-row. "
            f"Error was {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        outputs = []
        for row in rows:
            outputs.extend(generate_batch_with_fallback([row], tokenizer, model, torch, args))
        return outputs


def main() -> int:
    args = parse_args()
    if args.overwrite:
        for path in [args.out_jsonl, args.out_csv]:
            path_obj = Path(path)
            if path_obj.exists():
                path_obj.unlink()
    prompts = read_csv(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    done = set() if args.no_resume else existing_prompt_ids(args.out_jsonl)
    pending = [row for row in prompts if row["prompt_id"] not in done]
    print(f"Loaded {len(prompts)} prompts; {len(done)} already done; {len(pending)} pending.")
    if not pending:
        if Path(args.out_jsonl).exists():
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
        return 0

    if args.strict_row_seeds:
        args.batch_size = 1
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    tokenizer, model, torch = load_model(args.model, args.device)

    written = 0
    for start in range(0, len(pending), args.batch_size):
        batch_rows = pending[start : start + args.batch_size]
        outputs = generate_batch_with_fallback(batch_rows, tokenizer, model, torch, args)
        for row, (response, finish_reason) in zip(batch_rows, outputs):
            out_row = dict(row)
            out_row["generation_model"] = args.model
            out_row["response_text"] = response
            out_row["finish_reason"] = finish_reason
            append_jsonl(args.out_jsonl, out_row)
            written += 1
        if written % args.flush_every == 0:
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
            print(f"Wrote {written}/{len(pending)} pending generations.")

    write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
    print(f"Saved generations to {args.out_jsonl} and {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
