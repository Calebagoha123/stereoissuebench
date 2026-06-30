#!/usr/bin/env python3
"""Generate writing-assistance responses via the OpenAI API.

Drop-in alternative to 02_run_generation.py for a closed-model arm. Reads the
same prompt CSVs (system_text + prompt_text per row), writes the same
GENERATION_COLUMNS JSONL/CSV, and shares the resume/shard logic so a run can be
stopped and restarted without re-billing completed rows.

The model id is passed verbatim to the API (e.g. gpt-4o-mini, gpt-4.1-nano, or
whatever the current mini/nano slug is) -- confirm the exact slug on OpenAI's
model list. Reasoning models (gpt-5 family) reject sampling params, so pass
--reasoning-effort and omit --temperature for those.

Needs: openai>=1.0, OPENAI_API_KEY in the environment.
Cost note: this is synchronous/concurrent. For a ~50% discount on an offline run
use the Batch API instead (see --dump-batch-jsonl).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DEFAULT_RESULTS_DIR, GENERATION_COLUMNS
from io_utils import append_jsonl, existing_prompt_ids, read_csv, read_jsonl, write_csv
from shard_utils import select_shard

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **k):  # type: ignore
        return x if x is not None else iter(())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default=str(DEFAULT_RESULTS_DIR / "prompts_pilot.csv"))
    p.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "generations_openai.jsonl"))
    p.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "generations_openai.csv"))
    p.add_argument("--model", required=True, help="OpenAI model id, verbatim (e.g. gpt-4o-mini).")
    p.add_argument("--max-completion-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Ignored when --reasoning-effort is set (reasoning models reject it).")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--reasoning-effort",
                   choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                   help="Set for gpt-5-family reasoning models; disables sampling params. "
                        "Supported set varies by model (e.g. gpt-5.4-mini wants 'none', not 'minimal').")
    p.add_argument("--concurrency", type=int, default=8, help="Parallel in-flight requests.")
    p.add_argument("--limit", type=int)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume-from-jsonl", action="append", default=[])
    p.add_argument("--dump-batch-jsonl",
                   help="Instead of calling the API, write a Batch-API input .jsonl here and exit.")
    p.add_argument("--load-batch-output",
                   help="Parse an OpenAI Batch results .jsonl into the JSONL/CSV generation outputs "
                        "(joins metadata from --prompts on custom_id) and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report pending count + token/cost estimate and exit without calling the API.")
    return p.parse_args()


def build_request_body(row: dict, args: argparse.Namespace) -> dict:
    messages = []
    system = (row.get("system_text") or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": row["prompt_text"]})
    body: dict = {
        "model": args.model,
        "messages": messages,
        "max_completion_tokens": args.max_completion_tokens,
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    else:
        body["temperature"] = args.temperature
        body["top_p"] = args.top_p
    return body


def call_one(client, row: dict, args: argparse.Namespace) -> tuple[dict, str, str]:
    try:
        resp = client.chat.completions.create(**build_request_body(row, args))
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        return row, text, choice.finish_reason or "ok"
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
        return row, f"GENERATION_ERROR: {type(exc).__name__}: {exc}", "error"


def dump_batch(pending: list[dict], args: argparse.Namespace) -> None:
    out = Path(args.dump_batch_jsonl)
    with out.open("w") as fh:
        for row in pending:
            fh.write(json.dumps({
                "custom_id": row["prompt_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_request_body(row, args),
            }) + "\n")
    print(f"Wrote {len(pending)} Batch-API requests to {out}\n"
          f"Submit with: openai api batches.create -i {out} --endpoint /v1/chat/completions --completion-window 24h")


def load_batch_output(args: argparse.Namespace) -> None:
    """Reassemble OpenAI Batch results into the standard generation JSONL/CSV."""
    by_id = {r["prompt_id"]: r for r in read_csv(args.prompts)}
    n_ok = n_err = 0
    with Path(args.load_batch_output).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("custom_id")
            row = by_id.get(pid)
            if row is None:
                continue
            out_row = dict(row)
            out_row["generation_model"] = args.model
            err = rec.get("error")
            resp = rec.get("response") or {}
            body = resp.get("body") or {}
            if err or resp.get("status_code", 200) != 200 or not body.get("choices"):
                out_row["response_text"] = f"GENERATION_ERROR: {json.dumps(err) if err else resp.get('status_code')}"
                out_row["finish_reason"] = "error"
                n_err += 1
            else:
                choice = body["choices"][0]
                out_row["response_text"] = (choice.get("message", {}).get("content") or "").strip()
                out_row["finish_reason"] = choice.get("finish_reason") or "ok"
                n_ok += 1
            append_jsonl(args.out_jsonl, out_row)
    write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
    print(f"Loaded batch output: {n_ok} ok, {n_err} errors -> {args.out_jsonl} / {args.out_csv}")


def main() -> int:
    args = parse_args()
    if args.load_batch_output:
        load_batch_output(args)
        return 0
    if args.overwrite:
        for path in [args.out_jsonl, args.out_csv]:
            if Path(path).exists():
                Path(path).unlink()

    prompts = read_csv(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    total = len(prompts)
    prompts = select_shard(prompts, args.num_shards, args.shard_index)
    if args.num_shards > 1:
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(prompts)} of {total} rows.")

    done: set[str] = set()
    if not args.no_resume:
        done |= existing_prompt_ids(args.out_jsonl)
        for path in args.resume_from_jsonl:
            done |= existing_prompt_ids(path)
    pending = [r for r in prompts if r["prompt_id"] not in done]
    print(f"Loaded {len(prompts)} prompts; {len(done)} done; {len(pending)} pending.")
    if not pending:
        if Path(args.out_jsonl).exists():
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
        return 0

    if args.dry_run:
        print(f"[dry-run] would generate {len(pending)} rows on '{args.model}'. No API calls made.")
        return 0
    if args.dump_batch_jsonl:
        dump_batch(pending, args)
        return 0

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai>=1.0 in the run environment.") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment before running.")
    client = OpenAI()

    written = 0
    progress = tqdm(total=len(pending), desc=f"OpenAI:{args.model}", unit="row")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(call_one, client, row, args) for row in pending]
        for fut in as_completed(futures):
            row, response, finish_reason = fut.result()
            out_row = dict(row)
            out_row["generation_model"] = args.model
            out_row["response_text"] = response
            out_row["finish_reason"] = finish_reason
            append_jsonl(args.out_jsonl, out_row)
            written += 1
            if hasattr(progress, "update"):
                progress.update(1)
            if written % args.flush_every == 0:
                write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)

    write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
    print(f"Saved {written} generations to {args.out_jsonl} and {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
