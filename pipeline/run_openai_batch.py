#!/usr/bin/env python3
"""Submit an OpenAI Batch job safely: sequential shards + adaptive sizing.

Why this exists: the Batch API self-throttles, so per-minute (RPM/TPM) rate
limits are never hit. The ONE limit that can reject a job is the per-model
*enqueued-token* ceiling at submit time, which is tier-dependent. This runner
guarantees we never trip it:

  * shards the prompts and submits ONE shard at a time, waiting for each to reach
    a terminal state before the next -- so only one shard's tokens are ever queued;
  * if a shard is still rejected for a token/queue limit, it HALVES the shard and
    retries, self-tuning to whatever the account tier allows;
  * polls each batch to completion, downloads the output, and reassembles it into
    the standard GENERATION_COLUMNS JSONL/CSV (joining metadata on prompt_id);
  * is resumable -- prompt_ids already in --out-jsonl are skipped on restart.

Needs: openai>=1.0, OPENAI_API_KEY. Run from the pipeline/ dir.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

from config import GENERATION_COLUMNS
from io_utils import append_jsonl, existing_prompt_ids, read_csv, read_jsonl, write_csv

TERMINAL = {"completed", "failed", "expired", "cancelled"}

_TRANSIENT = {"APIConnectionError", "APITimeoutError", "InternalServerError",
              "RateLimitError", "RemoteProtocolError", "ReadError", "ReadTimeout",
              "ConnectError", "ConnectTimeout"}


def _is_transient(exc: Exception) -> bool:
    return type(exc).__name__ in _TRANSIENT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--reasoning-effort")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=1000)
    p.add_argument("--shard-requests", type=int, default=8000,
                   help="Initial requests per shard; auto-halved on a queue-limit rejection.")
    p.add_argument("--min-shard", type=int, default=250,
                   help="Give up sharding below this size (real error, not a queue limit).")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--limit", type=int, help="Only consider the first N prompt rows (for smoke tests).")
    return p.parse_args()


def body_for(row: dict, args) -> dict:
    msgs = []
    system = (row.get("system_text") or "").strip()
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": row["prompt_text"]})
    body = {"model": args.model, "messages": msgs,
            "max_completion_tokens": args.max_completion_tokens}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    else:
        body["temperature"] = args.temperature
        body["top_p"] = args.top_p
    return body


def is_queue_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("token_limit" in s or "enqueued" in s or "queue" in s
            or ("limit" in s and "token" in s))


def submit_shard(client, rows: list[dict], args) -> str:
    """Upload + create one batch, return its id. Raises on API error."""
    buf = io.BytesIO()
    for row in rows:
        line = json.dumps({"custom_id": row["_cid"], "method": "POST",
                           "url": "/v1/chat/completions", "body": body_for(row, args)})
        buf.write((line + "\n").encode())
    buf.seek(0)
    buf.name = "batch_shard.jsonl"
    f = client.files.create(file=buf, purpose="batch")
    batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    return batch.id


def poll(client, batch_id: str, poll_seconds: int):
    while True:
        try:
            b = client.batches.retrieve(batch_id)
        except Exception as exc:  # noqa: BLE001
            if _is_transient(exc):
                print(f"  poll retrieve failed ({type(exc).__name__}); retrying in {poll_seconds}s...", flush=True)
                time.sleep(poll_seconds)
                continue
            raise
        rc = b.request_counts
        print(f"  [{batch_id}] {b.status}  "
              f"{getattr(rc,'completed',0)}/{getattr(rc,'total',0)} done, "
              f"{getattr(rc,'failed',0)} failed", flush=True)
        if b.status in TERMINAL:
            return b
        time.sleep(poll_seconds)


def reassemble(client, batch, row_by_cid: dict, args, stats: dict) -> int:
    if not batch.output_file_id:
        print(f"  no output file (status={batch.status}); nothing reassembled.")
        return 0
    text = None
    for attempt in range(8):
        try:
            text = client.files.content(batch.output_file_id).text
            break
        except Exception as exc:  # noqa: BLE001
            if _is_transient(exc) and attempt < 7:
                print(f"  output download failed ({type(exc).__name__}); retry {attempt+1}/8...", flush=True)
                time.sleep(5)
                continue
            raise
    n = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        row = row_by_cid.get(rec.get("custom_id"))
        if row is None:
            continue
        out = dict(row)
        out.pop("_cid", None)
        out["generation_model"] = args.model
        resp = rec.get("response") or {}
        bd = resp.get("body") or {}
        if rec.get("error") or resp.get("status_code", 200) != 200 or not bd.get("choices"):
            out["response_text"] = f"GENERATION_ERROR: {rec.get('error') or resp.get('status_code')}"
            out["finish_reason"] = "error"
            stats["fr"]["error"] = stats["fr"].get("error", 0) + 1
        else:
            ch = bd["choices"][0]
            out["response_text"] = (ch.get("message", {}).get("content") or "").strip()
            out["finish_reason"] = ch.get("finish_reason") or "ok"
            usage = bd.get("usage") or {}
            stats["out"].append(usage.get("completion_tokens", 0))
            stats["in"].append(usage.get("prompt_tokens", 0))
            stats["fr"][out["finish_reason"]] = stats["fr"].get(out["finish_reason"], 0) + 1
        append_jsonl(args.out_jsonl, out)
        n += 1
    return n


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
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai>=1.0") from exc
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY.")
    client = OpenAI(max_retries=8)

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
        print(f"Submitting shard of {len(shard)} requests (rows {i}..{i+len(shard)}).")
        try:
            bid = submit_shard(client, shard, args)
        except Exception as exc:  # noqa: BLE001
            if is_queue_limit_error(exc) and shard_size > args.min_shard:
                shard_size = max(args.min_shard, shard_size // 2)
                print(f"  queue-limit rejection; halving shard to {shard_size} and retrying.")
                continue
            raise SystemExit(f"Batch submit failed (not a queue limit): {exc}")
        batch = poll(client, bid, args.poll_seconds)
        wrote = reassemble(client, batch, row_by_cid, args, stats)
        write_csv(args.out_csv, read_jsonl(args.out_jsonl), GENERATION_COLUMNS)
        print(f"  reassembled {wrote} rows -> {args.out_csv}")
        print_stats(stats)
        if batch.status != "completed":
            print(f"  shard ended {batch.status}; re-run to retry the remainder.")
        i += len(shard)

    print(f"Done. {len(existing_prompt_ids(args.out_jsonl))} total rows in {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
