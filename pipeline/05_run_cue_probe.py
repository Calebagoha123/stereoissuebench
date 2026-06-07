#!/usr/bin/env python3
"""Cue-legibility probe: ask the generation model to infer a user's profile.

Cue-only manipulation check. For each first-name cue ("My name is Latoya.") the
generation model is asked, in three separate prompts, to infer the user's race,
gender, and political leaning. Categorical recall against the intended subgroup
(plus the "Cannot tell" abstention rate) measures how legible each name is as a
demographic signal; the continuous political readout lands on the same [-1, +1]
liberal scale as the generation-side stance, so inferred lean can be compared to
the expressed stance shift and the CES subgroup mean.

Runs on the same model as ``02_run_generation.py`` (config.DEFAULT_GEN_MODEL).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

from config import DEFAULT_GEN_MODEL, DEFAULT_RESULTS_DIR, INPUT_DIR
from cues import name_cues_from_csv
from hf_utils import apply_chat_template, resolve_local_model_path
from io_utils import append_jsonl, existing_prompt_ids, read_jsonl, write_csv
from probe import PROBE_ATTRIBUTES, build_probe_prompt, parse_probe
from prompting import slugify, stable_seed
from shard_utils import select_shard

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is in the project requirements.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def write(self, message: str):
            print(message)


DEFAULT_NAMES_CSV = INPUT_DIR / "names" / "names_tzioumis.csv"

PROBE_COLUMNS = [
    "prompt_id",
    "source",
    "subgroup",
    "intended_race",
    "intended_gender",
    "name",
    "cue_text",
    "attribute",
    "repeat",
    "seed",
    "prompt_text",
    "probe_model",
    "response_text",
    "parsed_value",
    "finish_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default=str(DEFAULT_NAMES_CSV))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "cue_probe.jsonl"))
    parser.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "cue_probe.csv"))
    parser.add_argument("--model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def build_probe_rows(names_csv: str, repeats: int) -> list[dict]:
    cues = name_cues_from_csv(names_csv)
    rows: list[dict] = []
    for cue in cues:
        race, _, gender = cue.cue_group.partition("_")
        for attribute in PROBE_ATTRIBUTES:
            for repeat in range(1, repeats + 1):
                prompt_id = (
                    f"{slugify(cue.cue_value)}__{cue.cue_group}__"
                    f"{attribute}__r{repeat:02d}"
                )
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "source": "tzioumis",
                        "subgroup": cue.cue_group,
                        "intended_race": race,
                        "intended_gender": gender,
                        "name": cue.cue_value,
                        "cue_text": cue.cue_text,
                        "attribute": attribute,
                        "repeat": str(repeat),
                        "seed": str(stable_seed(prompt_id)),
                        "prompt_text": build_probe_prompt(cue.cue_text, attribute),
                    }
                )
    return rows


def load_model(model_path: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing probe dependencies. Install torch and transformers in the run environment."
        ) from exc

    resolved = resolve_local_model_path(model_path)
    print(f"Loading probe tokenizer from {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, padding_side="left", local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading probe model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved, torch_dtype=torch.bfloat16, device_map=device, local_files_only=True
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


def generate_batch(rows: list[dict], tokenizer, model, torch, args) -> list[str]:
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


def generate_batch_with_fallback(rows, tokenizer, model, torch, args) -> list[tuple[str, str]]:
    try:
        return [(text, "ok") for text in generate_batch(rows, tokenizer, model, torch, args)]
    except Exception as exc:
        if len(rows) == 1:
            return [(f"PROBE_ERROR: {type(exc).__name__}: {exc}", "error")]
        print(
            f"Batch probe failed for {len(rows)} rows; retrying row-by-row. "
            f"Error was {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        outputs: list[tuple[str, str]] = []
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

    rows = build_probe_rows(args.names, args.repeats)
    if args.limit is not None:
        rows = rows[: args.limit]
    total_rows = len(rows)
    rows = select_shard(rows, args.num_shards, args.shard_index)
    if args.num_shards > 1:
        print(f"Shard {args.shard_index}/{args.num_shards} selected {len(rows)} of {total_rows} rows.")

    done = set()
    if not args.no_resume:
        done |= existing_prompt_ids(args.out_jsonl)
    pending = [row for row in rows if row["prompt_id"] not in done]
    print(f"Built {len(rows)} probe rows; {len(done)} already done; {len(pending)} pending.")
    if not pending:
        if Path(args.out_jsonl).exists():
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), PROBE_COLUMNS)
        return 0

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    tokenizer, model, torch = load_model(args.model, args.device)

    written = 0
    progress = tqdm(
        total=len(pending), desc="Probing", unit="row",
        dynamic_ncols=True, disable=args.no_progress,
    )
    with progress:
        for start in range(0, len(pending), args.batch_size):
            batch_rows = pending[start : start + args.batch_size]
            outputs = generate_batch_with_fallback(batch_rows, tokenizer, model, torch, args)
            for row, (response, finish_reason) in zip(batch_rows, outputs):
                out_row = dict(row)
                out_row["probe_model"] = args.model
                out_row["response_text"] = response
                out_row["parsed_value"] = parse_probe(row["attribute"], response)
                out_row["finish_reason"] = finish_reason
                append_jsonl(args.out_jsonl, out_row)
                written += 1
            progress.update(len(batch_rows))
            progress.set_postfix(batch_size=len(batch_rows), written=written)
            if written % args.flush_every == 0:
                write_csv(args.out_csv, read_jsonl(args.out_jsonl), PROBE_COLUMNS)
                progress.write(f"Wrote {written}/{len(pending)} pending probes.")

    all_rows = read_jsonl(args.out_jsonl)
    write_csv(args.out_csv, all_rows, PROBE_COLUMNS)
    parse_errors = sum(1 for row in all_rows if str(row.get("parsed_value")) == "PARSE_ERROR")
    print(f"Saved probes to {args.out_jsonl} and {args.out_csv}")
    print(f"Parse errors: {parse_errors}/{len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
