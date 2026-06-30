#!/usr/bin/env python3
"""Behavioral belief probes A2 (opinion prediction) + A3 (attribute relevance).

A2: for every cue x CES issue, ask the model how likely the cued user is to
SUPPORT the policy (0-100). A3: for every attribute x issue, ask how predictive
that attribute is of opinion on the issue (0-100). Both reuse the cue strings,
issues, and liberal axis of the main generation run, so the model's *beliefs* can
be lined up against the stance it actually *writes* (the bert_eval_* outputs) and
against CES ground truth.

Runs on the same local HF model as 02/05 (config.DEFAULT_GEN_MODEL); for closed
models (GPT) the same rows can be sent through the OpenAI batch path — the prompt
builders in beliefs.py are model-agnostic.

    python pipeline/07_run_belief_probe.py --kind opinion --model <path> --device cuda:0
    python pipeline/07_run_belief_probe.py --kind relevance --model <path> --device cuda:0
    python pipeline/07_run_belief_probe.py --kind both --model <path> --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from beliefs import (
    RELEVANCE_ATTRIBUTES,
    build_opinion_prompt,
    build_relevance_prompt,
    parse_score,
)
from config import DEFAULT_GEN_MODEL, DEFAULT_ISSUES_CSV, DEFAULT_RESULTS_DIR, DEFAULT_WORDING_CSV
from cues import all_cues
from io_utils import append_jsonl, existing_prompt_ids, read_csv, read_jsonl, write_csv
from probe_runtime import generate_batch_with_fallback, load_model
from prompting import apply_issue_wording, main_issues, slugify, stable_seed
from shard_utils import select_shard

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **kwargs):  # type: ignore
        return x if x is not None else iter(())


BELIEF_COLUMNS = [
    "prompt_id", "probe_kind",
    # opinion-only
    "cue_condition", "cue_family", "cue_group", "cue_value", "cue_memory",
    "stance_target", "liberal_sign",
    # relevance-only
    "attribute", "attribute_phrase",
    # shared
    "issue_id", "topic", "repeat", "seed", "prompt_text",
    "probe_model", "response_text", "parsed_score", "finish_reason",
]


def load_issues() -> list[dict]:
    issues = apply_issue_wording(
        main_issues(read_csv(DEFAULT_ISSUES_CSV)), read_csv(DEFAULT_WORDING_CSV)
    )
    out = []
    for issue in issues:
        issue_id = issue.get("ces_variable", "").strip() or slugify(issue["topic_neutral"])
        out.append({
            "issue_id": issue_id,
            "stance_target": issue.get("stance_target", "").strip(),
            "topic_neutral": issue.get("topic_neutral", "").strip(),
            "liberal_sign": issue.get("liberal_sign", "").strip(),
        })
    return out


def build_opinion_rows(issues: list[dict], repeats: int) -> list[dict]:
    rows = []
    for cue in all_cues():
        for issue in issues:
            policy = issue["stance_target"]
            for repeat in range(1, repeats + 1):
                prompt_id = f"opinion__{cue.cue_condition}__{issue['issue_id']}__r{repeat:02d}"
                rows.append({
                    "prompt_id": prompt_id, "probe_kind": "opinion",
                    "cue_condition": cue.cue_condition, "cue_family": cue.cue_family,
                    "cue_group": cue.cue_group, "cue_value": cue.cue_value,
                    "cue_memory": cue.cue_memory,
                    "stance_target": policy, "liberal_sign": issue["liberal_sign"],
                    "attribute": "", "attribute_phrase": "",
                    "issue_id": issue["issue_id"], "topic": issue["topic_neutral"],
                    "repeat": str(repeat), "seed": str(stable_seed(prompt_id)),
                    "prompt_text": build_opinion_prompt(cue.cue_memory, policy),
                })
    return rows


def build_relevance_rows(issues: list[dict], repeats: int) -> list[dict]:
    rows = []
    for attribute, phrase in RELEVANCE_ATTRIBUTES:
        for issue in issues:
            topic = issue["topic_neutral"]
            for repeat in range(1, repeats + 1):
                prompt_id = f"relevance__{attribute}__{issue['issue_id']}__r{repeat:02d}"
                rows.append({
                    "prompt_id": prompt_id, "probe_kind": "relevance",
                    "cue_condition": "", "cue_family": "", "cue_group": "",
                    "cue_value": "", "cue_memory": "", "stance_target": "", "liberal_sign": "",
                    "attribute": attribute, "attribute_phrase": phrase,
                    "issue_id": issue["issue_id"], "topic": topic,
                    "repeat": str(repeat), "seed": str(stable_seed(prompt_id)),
                    "prompt_text": build_relevance_prompt(phrase, topic),
                })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["opinion", "relevance", "both"], default="both")
    p.add_argument("--model", default=DEFAULT_GEN_MODEL)
    p.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "belief_probe.jsonl"))
    p.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "belief_probe.csv"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--flush-every", type=int, default=200)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.overwrite:
        for path in [args.out_jsonl, args.out_csv]:
            if Path(path).exists():
                Path(path).unlink()

    issues = load_issues()
    rows: list[dict] = []
    if args.kind in ("opinion", "both"):
        rows += build_opinion_rows(issues, args.repeats)
    if args.kind in ("relevance", "both"):
        rows += build_relevance_rows(issues, args.repeats)
    if args.limit is not None:
        rows = rows[: args.limit]

    total = len(rows)
    rows = select_shard(rows, args.num_shards, args.shard_index)
    if args.num_shards > 1:
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(rows)} of {total} rows.")

    done = set() if args.no_resume else existing_prompt_ids(args.out_jsonl)
    pending = [r for r in rows if r["prompt_id"] not in done]
    print(f"Built {len(rows)} belief rows; {len(done)} done; {len(pending)} pending.")
    if not pending:
        if Path(args.out_jsonl).exists():
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), BELIEF_COLUMNS)
        return 0

    tokenizer, model, torch, input_device = load_model(args.model, args.device)
    args.device = input_device  # generate_batch sends inputs here (resolved for device_map=auto)
    written = 0
    bar = tqdm(total=len(pending), desc=f"belief:{args.kind}", unit="row", disable=args.no_progress)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        outputs = generate_batch_with_fallback(batch, tokenizer, model, torch, args)
        for row, (response, finish_reason) in zip(batch, outputs):
            out = dict(row)
            out["probe_model"] = args.model
            out["response_text"] = response
            out["parsed_score"] = parse_score(response)
            out["finish_reason"] = finish_reason
            append_jsonl(args.out_jsonl, out)
            written += 1
        if hasattr(bar, "update"):
            bar.update(len(batch))
        if written % args.flush_every == 0:
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), BELIEF_COLUMNS)

    all_rows = read_jsonl(args.out_jsonl)
    write_csv(args.out_csv, all_rows, BELIEF_COLUMNS)
    errs = sum(1 for r in all_rows if str(r.get("parsed_score")) == "PARSE_ERROR")
    print(f"Saved {len(all_rows)} belief probes -> {args.out_csv}  (parse errors: {errs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
