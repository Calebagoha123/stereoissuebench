#!/usr/bin/env python3
"""Score the FULL generation corpus with an LLM-as-judge via the OpenAI Batch API.

Re-scores every response in the full_3x corpus with the classifier of record
(GPT-5.6 luna, chosen in the 2026-07-25 gold bake-off) using the identical 0-100
codebook prompt as the DeBERTa-comparable judge. Reads generation JSONL files
(prompt_id, stance_target, response_text, generation_model, generation_repeat,
arm, liberal_sign, ...) and writes a slim scored file that slots into the pipeline
in place of bert_pred_stance.

Design (mirrors pipeline/run_openai_batch.py): sequential shards, one batch at a
time, adaptive halving on a queue/size rejection, resumable. Resume + key-swap:
results append to <out>.jsonl keyed on a stable per-response key; a restart skips
keys already scored, so when a shard-submit hits the OpenAI credit limit you top
up / drop a new key into .env and re-run -- it continues from the first unscored
response. .env is reloaded every run.

Usage (from repo root):
  python stance_model/judge_corpus.py --model gpt-5.6-luna --reasoning-effort none \
    --out results/full_3x/luna_eval_all \
    --gen 'results/full_3x/gen_sonnet5_arm_*.jsonl' \
    --gen 'results/full_3x/gen_gpt56terra_arm_*.jsonl' \
    --gen 'data/processed/full_3x_local/gen_*_arm_*.jsonl'
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stance_model"))
from judge_batch import PROMPT, parse_score, load_env  # noqa: E402

# metadata carried through to the scored output (whatever the gen row has)
CARRY = ["prompt_id", "arm", "cue_condition", "cue_family", "cue_group", "issue_id",
         "ces_variable", "stance_target", "liberal_sign", "generation_model",
         "generation_repeat", "instance_id", "finish_reason"]
_TRANSIENT = {"APIConnectionError", "APITimeoutError", "InternalServerError",
              "RateLimitError", "RemoteProtocolError", "ReadError", "ReadTimeout",
              "ConnectError", "ConnectTimeout"}


def is_transient(e):
    return type(e).__name__ in _TRANSIENT


def is_queue_limit(e):
    s = str(e).lower()
    return ("token_limit" in s or "enqueued" in s or "queue" in s
            or ("limit" in s and "token" in s)
            or "413" in s or "too large" in s or "size" in s or "256" in s)  # batch-file size cap


def is_credit_limit(e):
    s = str(e).lower()
    return ("insufficient_quota" in s or "billing" in s or "hard_limit" in s
            or "exceeded your current quota" in s or "credit" in s)


def row_key(r):
    return "|".join(str(r.get(k, "")) for k in
                    ("generation_model", "prompt_id", "generation_repeat", "arm"))


def load_gen_rows(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def done_keys(jsonl_path):
    p = Path(jsonl_path)
    done = set()
    if p.exists():
        with p.open() as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("judge_score") not in (None, ""):
                        done.add(rec["_key"])
    return done


def rewrite_csv(jsonl_path, csv_path):
    import csv
    seen, best = set(), {}
    with open(jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            k = rec["_key"]
            # prefer a scored record over a blank one
            if k not in best or (best[k].get("judge_score") in (None, "") and rec.get("judge_score") not in (None, "")):
                best[k] = rec
    cols = CARRY + ["judge_model", "judge_raw", "judge_score"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in best.values():
            w.writerow(rec)
    return len(best)


def submit_shard(client, rows, args):
    buf = io.BytesIO()
    for row in rows:
        prompt = PROMPT.format(proposition=str(row.get("stance_target", "")).strip(),
                               response_text=str(row.get("response_text", "")))
        body = {"model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": args.max_completion_tokens}
        if args.reasoning_effort:
            body["reasoning_effort"] = args.reasoning_effort
        else:
            body["temperature"] = 0
        buf.write((json.dumps({"custom_id": row["_cid"], "method": "POST",
                               "url": "/v1/chat/completions", "body": body}) + "\n").encode())
    buf.seek(0)
    buf.name = "corpus_shard.jsonl"
    f = client.files.create(file=buf, purpose="batch")
    return client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                 completion_window="24h").id


def poll(client, bid, poll_seconds):
    while True:
        try:
            b = client.batches.retrieve(bid)
        except Exception as e:  # noqa: BLE001
            if is_transient(e):
                print(f"  poll failed ({type(e).__name__}); retry in {poll_seconds}s", flush=True)
                time.sleep(poll_seconds); continue
            raise
        rc = b.request_counts
        print(f"  [{bid}] {b.status}: {getattr(rc,'completed',0)}/{getattr(rc,'total',0)} done, "
              f"{getattr(rc,'failed',0)} failed", flush=True)
        if b.status in {"completed", "failed", "expired", "cancelled"}:
            return b
        time.sleep(poll_seconds)


def reassemble(client, batch, row_by_cid, args):
    n_ok = 0
    if not batch.output_file_id:
        return 0
    text = client.files.content(batch.output_file_id).text
    with open(args.out + ".jsonl", "a") as out:
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            row = row_by_cid.get(rec.get("custom_id"))
            if row is None:
                continue
            resp = rec.get("response") or {}
            bd = resp.get("body") or {}
            if rec.get("error") or resp.get("status_code", 200) != 200 or not bd.get("choices"):
                raw, score = f"ERROR:{rec.get('error') or resp.get('status_code')}", None
            else:
                raw = (bd["choices"][0].get("message", {}).get("content") or "").strip()
                score = parse_score(raw)
            out_row = {k: row.get(k, "") for k in CARRY}
            out_row["_key"] = row["_key"]
            out_row["judge_model"] = args.model
            out_row["judge_raw"] = raw
            out_row["judge_score"] = "" if score is None else score
            out.write(json.dumps(out_row) + "\n")
            if score is not None:
                n_ok += 1
    return n_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", required=True, help="Glob(s) of gen JSONL files. Repeatable.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="Output path prefix (writes <out>.jsonl and <out>.csv).")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--max-completion-tokens", type=int, default=2000)
    ap.add_argument("--shard-requests", type=int, default=20000)
    ap.add_argument("--min-shard", type=int, default=250)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    load_env()
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (repo .env).")
    from openai import OpenAI
    client = OpenAI(max_retries=8)

    rows = load_gen_rows(args.gen)
    if args.limit:
        rows = rows[:args.limit]
    for i, r in enumerate(rows):
        r["_cid"] = f"i{i}"
        r["_key"] = row_key(r)
    row_by_cid = {r["_cid"]: r for r in rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(args.out + ".jsonl")
    pending = [r for r in rows if r["_key"] not in done]
    print(f"{len(rows)} corpus responses; {len(done)} already scored; {len(pending)} pending.", flush=True)
    if not pending:
        n = rewrite_csv(args.out + ".jsonl", args.out + ".csv")
        print(f"Nothing to do. {n} rows in {args.out}.csv")
        return 0

    i, shard_size = 0, args.shard_requests
    while i < len(pending):
        shard = pending[i:i + shard_size]
        print(f"Submitting shard {len(shard)} (rows {i}..{i+len(shard)} of {len(pending)} pending)", flush=True)
        try:
            bid = submit_shard(client, shard, args)
        except Exception as e:  # noqa: BLE001
            if is_credit_limit(e):
                print(f"\n*** OpenAI CREDIT LIMIT hit: {e}\n"
                      f"*** {len(done)+i} scored so far. Top up / put a new key in .env and RE-RUN "
                      f"this same command to resume.", flush=True)
                rewrite_csv(args.out + ".jsonl", args.out + ".csv")
                return 2
            if is_queue_limit(e) and shard_size > args.min_shard:
                shard_size = max(args.min_shard, shard_size // 2)
                print(f"  queue-limit; halving shard to {shard_size}", flush=True)
                continue
            raise SystemExit(f"Submit failed (not credit/queue): {e}")
        batch = poll(client, bid, args.poll_seconds)
        ok = reassemble(client, batch, row_by_cid, args)
        n = rewrite_csv(args.out + ".jsonl", args.out + ".csv")
        print(f"  shard done: {ok} newly scored; {n} total in {args.out}.csv", flush=True)
        if batch.status != "completed":
            print(f"  shard ended {batch.status}; re-run to retry remainder.", flush=True)
        i += len(shard)

    n = rewrite_csv(args.out + ".jsonl", args.out + ".csv")
    print(f"Done. {n} rows scored -> {args.out}.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
