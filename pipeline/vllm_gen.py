#!/usr/bin/env python3
"""Fast local generation with vLLM (continuous batching).

Drop-in replacement for 02_run_generation.py's output: same prompt formatting
(hf_utils.apply_chat_template, enable_thinking=False), same per-row seeding
(sha256 of prompt_id:seed), same GENERATION_COLUMNS jsonl+csv schema, resume
support. Runs under the isolated vLLM venv, e.g.:

    /data/<user>/vllm-venv/bin/python pipeline/vllm_gen.py \
        --prompts data/processed/full_3x/prompts_arm_a.csv \
        --model /data/resource/huggingface/hub/models--google--gemma-3-12b-it \
        --out-jsonl OUT/gen_gemma_arm_a.jsonl --out-csv OUT/gen_gemma_arm_a.csv \
        --max-new-tokens 2000
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from config import GENERATION_COLUMNS
from hf_utils import apply_chat_template, resolve_local_model_path
from io_utils import append_jsonl, existing_prompt_ids, read_csv, read_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-input-tokens", type=int, default=768)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--chunk", type=int, default=4000, help="Rows per generate() / write checkpoint.")
    p.add_argument("--limit", type=int)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def row_seed(row: dict) -> int:
    src = f"{row['prompt_id']}:{row.get('seed', '')}".encode("utf-8")
    return int(hashlib.sha256(src).hexdigest()[:8], 16)


def main() -> int:
    args = parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    prompts = read_csv(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    done: set[str] = set()
    if not args.no_resume:
        done = existing_prompt_ids(args.out_jsonl)
    pending = [r for r in prompts if r["prompt_id"] not in done]
    print(f"Loaded {len(prompts)} prompts; {len(done)} already done; {len(pending)} pending.", flush=True)
    if not pending:
        if Path(args.out_jsonl).exists():
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
        return 0

    resolved = resolve_local_model_path(args.model)
    print(f"Loading tokenizer + vLLM engine from {resolved}", flush=True)
    tok = AutoTokenizer.from_pretrained(resolved, local_files_only=True)

    llm = LLM(
        model=resolved,
        tokenizer=resolved,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        seed=0,
    )

    written = 0
    for start in range(0, len(pending), args.chunk):
        chunk = pending[start : start + args.chunk]
        token_prompts = []
        sampling = []
        for row in chunk:
            formatted = apply_chat_template(tok, row["prompt_text"], row.get("system_text", ""))
            ids = tok(
                formatted,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_input_tokens,
            ).input_ids
            token_prompts.append(TokensPrompt(prompt_token_ids=ids))
            sampling.append(
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_new_tokens,
                    seed=row_seed(row),
                )
            )

        outputs = llm.generate(token_prompts, sampling)
        for row, out in zip(chunk, outputs):
            comp = out.outputs[0]
            rec = dict(row)
            rec["generation_model"] = args.model
            rec["response_text"] = comp.text.strip()
            rec["finish_reason"] = comp.finish_reason  # "stop" | "length" | ...
            append_jsonl(args.out_jsonl, rec)
            written += 1
        write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
        print(f"Wrote {written}/{len(pending)} pending generations.", flush=True)

    print(f"Saved generations to {args.out_jsonl} and {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
