#!/usr/bin/env python3
"""Submit an Anthropic Message Batches job: sequential shards + adaptive sizing.

Mirror of run_openai_batch.py for a closed Anthropic-model arm. Reads the same
prompt CSVs (system_text + prompt_text per row), writes the same
GENERATION_COLUMNS JSONL/CSV (joined on prompt_id), and shares the
resume/shard/reassemble design so a run can be stopped and restarted without
re-billing completed rows.

Why sharding: the Batches API self-throttles, so per-minute limits are never
hit. The binding limit is per-batch: <=100,000 requests OR <=256 MB of request
body. This runner submits ONE shard at a time, waiting for each to reach a
terminal state before the next, and HALVES the shard on a too-large / too-many
rejection -- self-tuning to whatever fits.

custom_id note: Anthropic requires custom_id to match ^[a-zA-Z0-9_-]{1,64}$.
Some full_3x arm-B prompt_ids are 67 chars, so we key requests on a stable
row index ("i<n>") and map back to prompt_id at reassembly time.

Thinking / sampling note: Claude Sonnet 5 runs adaptive thinking by default and
REJECTS temperature/top_p (400). To mirror the OpenAI "no reasoning" arm we pass
thinking={"type": "disabled"} and send no sampling params.

Needs: anthropic>=0.40, ANTHROPIC_API_KEY. Run from the pipeline/ dir.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import GENERATION_COLUMNS
from io_utils import append_jsonl, existing_prompt_ids, read_csv, read_jsonl, write_csv

_TRANSIENT = {"APIConnectionError", "APITimeoutError", "InternalServerError",
              "RemoteProtocolError", "ReadError", "ReadTimeout", "ConnectError",
              "ConnectTimeout", "OverloadedError"}


def _is_transient(exc: Exception) -> bool:
    return type(exc).__name__ in _TRANSIENT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--model", required=True, help="Anthropic model id, verbatim (e.g. claude-sonnet-5).")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--thinking", choices=["disabled", "adaptive"], default="disabled",
                   help="'disabled' mirrors the no-reasoning arm; 'adaptive' turns thinking on.")
    p.add_argument("--shard-requests", type=int, default=5000,
                   help="Initial requests per shard; auto-halved on a too-large/too-many rejection. "
                        "Kept modest so each results() stream stays small (mid-stream drops re-stream).")
    p.add_argument("--min-shard", type=int, default=250,
                   help="Give up sharding below this size (real error, not a size limit).")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--limit", type=int, help="Only consider the first N prompt rows (for smoke tests).")
    return p.parse_args()


def params_for(row: dict, args) -> dict:
    body = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": row["prompt_text"]}],
        "thinking": {"type": args.thinking},
    }
    system = (row.get("system_text") or "").strip()
    if system:
        body["system"] = system
    return body


def is_size_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("request_too_large" in s or "413" in s or "too large" in s
            or "too many" in s or "256" in s or "100000" in s or "100,000" in s)


def submit_shard(client, rows: list[dict], args) -> str:
    """Create one batch, return its id. Raises on API error."""
    requests = [{"custom_id": row["_cid"], "params": params_for(row, args)} for row in rows]
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll(client, batch_id: str, poll_seconds: int):
    while True:
        try:
            b = client.messages.batches.retrieve(batch_id)
        except Exception as exc:  # noqa: BLE001
            if _is_transient(exc):
                print(f"  poll retrieve failed ({type(exc).__name__}); retrying in {poll_seconds}s...", flush=True)
                time.sleep(poll_seconds)
                continue
            raise
        rc = b.request_counts
        print(f"  [{batch_id}] {b.processing_status}  "
              f"{getattr(rc, 'succeeded', 0)} ok, {getattr(rc, 'errored', 0)} err, "
              f"{getattr(rc, 'processing', 0)} processing", flush=True)
        if b.processing_status == "ended":
            return b
        time.sleep(poll_seconds)


def reassemble(client, batch, row_by_cid: dict, args, stats: dict) -> int:
    """Stream results into the JSONL. Resilient to mid-stream connection drops:
    the results() stream can be re-fetched (it returns all results each call), so
    on a network error we re-stream and skip prompt_ids already written."""
    import httpx
    import anthropic
    conn_errs = (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout,
                 httpx.ConnectError, httpx.RemoteProtocolError, anthropic.APIConnectionError)
    already = existing_prompt_ids(args.out_jsonl)  # retry/resume-safe within a shard
    n = 0
    no_progress = 0
    while True:
        made = 0
        try:
            for result in client.messages.batches.results(batch.id):
                row = row_by_cid.get(result.custom_id)
                if row is None:
                    continue
                pid = row["prompt_id"]
                if pid in already:
                    continue
                out = dict(row)
                out.pop("_cid", None)
                out["generation_model"] = args.model
                res = result.result
                if res.type == "succeeded":
                    msg = res.message
                    text = "".join(b.text for b in msg.content if b.type == "text").strip()
                    out["response_text"] = text
                    out["finish_reason"] = msg.stop_reason or "ok"
                    stats["out"].append(msg.usage.output_tokens)
                    stats["in"].append(msg.usage.input_tokens)
                    stats["fr"][out["finish_reason"]] = stats["fr"].get(out["finish_reason"], 0) + 1
                else:  # errored | canceled | expired
                    err = getattr(res, "error", None)
                    out["response_text"] = f"GENERATION_ERROR: {res.type}: {err}"
                    out["finish_reason"] = "error"
                    stats["fr"]["error"] = stats["fr"].get("error", 0) + 1
                append_jsonl(args.out_jsonl, out)
                already.add(pid)
                n += 1
                made += 1
            return n
        except conn_errs as exc:
            no_progress = 0 if made else no_progress + 1
            if no_progress > 6:
                raise
            print(f"  results stream dropped ({type(exc).__name__}); re-streaming, "
                  f"{n} saved so far (retry {no_progress}/6)...", flush=True)
            time.sleep(5)


def print_stats(stats: dict) -> None:
    import statistics
    o = stats["out"]
    if not o:
        print(f"  usage: no successful responses | finish_reasons={dict(stats['fr'])}")
        return
    s = sorted(o)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    print(f"  output tokens: n={len(o)} mean={statistics.mean(o):.0f} "
          f"median={statistics.median(o):.0f} p90={p(.90)} p99={p(.99)} max={max(o)}")
    print(f"  totals: input={sum(stats['in'])} output={sum(o)} | "
          f"finish_reasons={dict(stats['fr'])}")


def main() -> int:
    args = parse_args()
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("pip install anthropic>=0.40") from exc
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY.")
    client = anthropic.Anthropic(max_retries=8)

    prompts = read_csv(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    for i, r in enumerate(prompts):
        r["_cid"] = f"i{i}"
    row_by_cid = {r["_cid"]: r for r in prompts}

    done = existing_prompt_ids(args.out_jsonl)
    pending = [r for r in prompts if r["prompt_id"] not in done]
    print(f"{len(prompts)} prompts; {len(done)} already done; {len(pending)} pending.")

    stats = {"out": [], "in": [], "fr": {}}
    i = 0
    shard_size = args.shard_requests
    while i < len(pending):
        shard = pending[i:i + shard_size]
        print(f"Submitting shard of {len(shard)} requests (rows {i}..{i + len(shard)}).")
        try:
            bid = submit_shard(client, shard, args)
        except Exception as exc:  # noqa: BLE001
            if is_size_limit_error(exc) and shard_size > args.min_shard:
                shard_size = max(args.min_shard, shard_size // 2)
                print(f"  size-limit rejection; halving shard to {shard_size} and retrying.")
                continue
            raise SystemExit(f"Batch submit failed (not a size limit): {exc}")
        batch = poll(client, bid, args.poll_seconds)
        wrote = reassemble(client, batch, row_by_cid, args, stats)
        write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
        print(f"  reassembled {wrote} rows -> {args.out_csv}")
        print_stats(stats)
        i += len(shard)

    print(f"Done. {len(existing_prompt_ids(args.out_jsonl))} total rows in {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
