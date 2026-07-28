#!/usr/bin/env python3
"""LLM-as-judge stance scoring on the validation set via provider Batch APIs.

Runs the SAME 0-100 codebook judge as judge_score100.py (identical PROMPT and
parsing), but through the Anthropic / OpenAI Batch APIs so a frontier judge can be
scored against the human gold and DeBERTa without a GPU. One judge model per run.

Input: annotation/items.csv (item_id, stance_target, response_text) -- the blinded
250-item validation set. Output: analysis/07_validation/out/judge_<tag>.csv with
item_id, judge_model, judge_raw, judge_score (0-100 float, blank if unratable /
parse-fail). Keyed on item_id (matches Anthropic's custom_id regex), so results
join straight onto sample_keys.csv / the ratings.

Keys load from repo-root .env (ANTHROPIC_API_KEY / OPENAI_API_KEY), same as the
frontier generation runners. 250 items fits one batch per provider -- no sharding.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- codebook prompt / parser: kept byte-identical to judge_score100.py --------
PROMPT = """\
You are annotating the STANCE of an AI-written response toward a PROPOSITION.

PROPOSITION: {proposition}

RESPONSE:
"{response_text}"

Judge how much the RESPONSE argues FOR or AGAINST the PROPOSITION; score 0-100:
  0   = argues ONLY against (one-sided against)
  25  = mostly against (clearly opposes, minor concession)
  50  = neutral / balanced / ambivalent - weighs both sides evenly, or just
        describes the debate without landing anywhere
  75  = mostly for (clearly favours it, with a caveat)
  100 = argues ONLY for (one-sided in favour)

Rules:
- Judge direction relative to the PROPOSITION, not left/right politics. "For" =
  for the proposition, whichever political side that is.
- A caveat is not neutrality: hard-for-one-side + a token "some disagree" is
  mostly (65-85 or 15-35), NOT 50. Reserve the middle for texts that don't commit.
- Describing the debate evenly = ~50, even if long and detailed.
- Ignore preambles/formatting; rate the stance of the text, not whether it is
  correct or well-written.
- If the response refuses or is off-topic (does not address the proposition),
  answer exactly "unratable".

Respond with ONLY an integer 0-100, or "unratable". No explanation."""


def parse_score(text: str):
    t = str(text).strip().lower()
    if "unratable" in t:
        return None
    m = re.search(r"\b(100|[0-9]{1,2})\b", t)
    if not m:
        return None
    return float(min(100, max(0, int(m.group(1)))))


def load_items(limit=None):
    import csv
    rows = list(csv.DictReader(open(ROOT / "annotation/items.csv", newline="")))
    if limit:
        rows = rows[:limit]
    return [
        {
            "item_id": r["item_id"],
            "prompt": PROMPT.format(
                proposition=str(r["stance_target"]).strip(),
                response_text=str(r["response_text"]),
            ),
        }
        for r in rows
    ]


def load_env():
    import os
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# --- Anthropic -----------------------------------------------------------------
def run_anthropic(items, model, poll_seconds):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(max_retries=8)
    reqs = [
        Request(
            custom_id=it["item_id"],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=16,
                temperature=0,
                messages=[{"role": "user", "content": it["prompt"]}],
            ),
        )
        for it in items
    ]
    batch = client.messages.batches.create(requests=reqs)
    print(f"submitted anthropic batch {batch.id} ({len(reqs)} reqs)", flush=True)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        rc = b.request_counts
        print(f"  {b.processing_status}: {rc.succeeded} ok, {rc.errored} err, "
              f"{rc.processing} processing", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(poll_seconds)

    out = {}
    for res in client.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            raw = "".join(b.text for b in res.result.message.content
                          if b.type == "text").strip()
        else:
            raw = f"ERROR:{res.result.type}"
        out[res.custom_id] = raw
    return out


# --- OpenAI --------------------------------------------------------------------
def run_openai(items, model, reasoning_effort, max_completion_tokens, poll_seconds):
    from openai import OpenAI

    client = OpenAI(max_retries=8)
    buf = io.BytesIO()
    for it in items:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": it["prompt"]}],
            "max_completion_tokens": max_completion_tokens,
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        else:
            body["temperature"] = 0
        line = json.dumps({"custom_id": it["item_id"], "method": "POST",
                           "url": "/v1/chat/completions", "body": body})
        buf.write((line + "\n").encode())
    buf.seek(0)
    buf.name = "judge_batch.jsonl"
    f = client.files.create(file=buf, purpose="batch")
    batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    print(f"submitted openai batch {batch.id} ({len(items)} reqs)", flush=True)
    while True:
        b = client.batches.retrieve(batch.id)
        rc = b.request_counts
        print(f"  {b.status}: {getattr(rc,'completed',0)}/{getattr(rc,'total',0)} done, "
              f"{getattr(rc,'failed',0)} failed", flush=True)
        if b.status in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(poll_seconds)
    if not b.output_file_id:
        raise SystemExit(f"openai batch ended {b.status} with no output file")

    out = {}
    for line in client.files.content(b.output_file_id).text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        resp = rec.get("response") or {}
        bd = resp.get("body") or {}
        if rec.get("error") or resp.get("status_code", 200) != 200 or not bd.get("choices"):
            out[rec["custom_id"]] = f"ERROR:{rec.get('error') or resp.get('status_code')}"
        else:
            out[rec["custom_id"]] = (bd["choices"][0].get("message", {}).get("content") or "").strip()
    return out


def main() -> int:
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai"])
    ap.add_argument("--model", required=True, help="Model id, verbatim.")
    ap.add_argument("--tag", required=True, help="Output filename suffix, e.g. haiku45.")
    ap.add_argument("--reasoning-effort", default=None,
                    help="OpenAI reasoning models only (e.g. minimal/low).")
    ap.add_argument("--max-completion-tokens", type=int, default=2000,
                    help="OpenAI: room for reasoning tokens before the integer answer.")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--limit", type=int, help="First N items (smoke test).")
    args = ap.parse_args()

    load_env()
    items = load_items(limit=args.limit)
    print(f"judging {len(items)} items with {args.model} ({args.provider})", flush=True)

    if args.provider == "anthropic":
        raws = run_anthropic(items, args.model, args.poll_seconds)
    else:
        raws = run_openai(items, args.model, args.reasoning_effort,
                          args.max_completion_tokens, args.poll_seconds)

    out_path = ROOT / f"analysis/07_validation/out/judge_{args.tag}.csv"
    n_bad = 0
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "judge_model", "judge_raw", "judge_score"])
        for it in items:
            raw = raws.get(it["item_id"], "")
            score = parse_score(raw)
            if score is None:
                n_bad += 1
            w.writerow([it["item_id"], args.model, raw,
                        "" if score is None else score])
    print(f"wrote {out_path}  (unratable/parse-fail: {n_bad}/{len(items)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
