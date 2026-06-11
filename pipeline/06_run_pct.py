#!/usr/bin/env python3
"""Political Compass Test arm: run the PCT instrument under baseline + name cues.

Two arms over the same 62-item instrument (see ``pct.py``):
  - baseline: the bare forced-choice Likert prompt, no cue.
  - implicit-demographic cue: each item prepended with a first-name persona line
    ("My name is X.") — the same cue strings the main generation run uses.

Grid = items x (baseline + name cues) x repeats. With the default 12 generation
names and 3 repeats that is 62 x 13 x 3 = 2,418 rows. Runs on the same local
model as ``02_run_generation.py`` (config.DEFAULT_GEN_MODEL); the generation
loop, batching, deterministic per-row seeds, resume and sharding all mirror that
stage and the cue probe. Output is one row per (item, cue, repeat) with the
parsed letter and an axis-aware ``liberal_score`` (+1 liberal, -1 conservative).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from dataclasses import replace
from pathlib import Path

from config import (
    DEFAULT_GEN_MODEL,
    DEFAULT_NAMES_GEN_CSV,
    DEFAULT_PCT_CSV,
    DEFAULT_RESULTS_DIR,
    PCT_COLUMNS,
)
from cues import Cue, all_cues, name_cues_from_csv
from hf_utils import apply_chat_template, resolve_local_model_path
from io_utils import append_jsonl, existing_prompt_ids, read_jsonl, write_csv
from pct import build_pct_prompt, load_pct_items, parse_pct_letter, score_letter
from prompting import stable_seed
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default=str(DEFAULT_PCT_CSV))
    parser.add_argument("--names", default=str(DEFAULT_NAMES_GEN_CSV))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "pct.jsonl"))
    parser.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "pct.csv"))
    parser.add_argument("--model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cue-set",
        choices=["names", "all"],
        default="names",
        help="'names' (default): no-cue baseline + the 12 implicit-demographic "
        "name cues. 'all': the full 29-cue grid from cues.all_cues() — baseline + "
        "explicit political + implicit political (states) + explicit demographic + "
        "implicit demographic (the same 12 names). Use 'all' to get the explicit "
        "manipulation check that makes the name effect interpretable.",
    )
    parser.add_argument(
        "--preamble-style",
        choices=["declarative", "paper"],
        default="declarative",
        help="Cue wording. 'declarative' (default): the bare self-identifications "
        "used across the rest of the pipeline ('I am a Republican.', 'I am a Black "
        "man.', 'My name is Jamal.', 'I live in Texas.'). 'paper': the request "
        "framing from Tornberg & Schimmel (arXiv:2604.27633) applied to ALL "
        "families so framing is held constant and only identity content varies "
        "('As a conservative Republican, …', 'As a Black man, …', 'As someone "
        "named Jamal, …', 'As someone who lives in Texas, I want your view on the "
        "following question.'). Write paper-style runs to a SEPARATE output file "
        "(the style is recorded in prompt_id and a preamble_style column).",
    )
    parser.add_argument(
        "--no-cue-only",
        action="store_true",
        help="Run only the no-cue baseline arm (overrides --cue-set).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def build_cues(cue_set: str, names_csv: str, no_cue_only: bool) -> list[Cue]:
    """Select the cue grid for the PCT arm.

    - ``no_cue_only``: just the no-cue baseline.
    - ``all``: the full 29-cue grid (``cues.all_cues()``; baseline first), whose
      name cues are the same 12 generation names.
    - ``names`` (default): baseline + the implicit-demographic name cues loaded
      from ``names_csv``.
    """
    baseline = Cue("baseline", "baseline", "none", "", "baseline")
    if no_cue_only:
        return [baseline]
    if cue_set == "all":
        return all_cues()
    return [baseline] + name_cues_from_csv(names_csv)


# Paper request framing (Tornberg & Schimmel, arXiv:2604.27633): every cue is
# rewritten as "As <identity>, I want your view on the following question." so the
# framing is held constant and only the identity *content* varies across families.
# The explicit-political identities keep the paper's verbatim ideological modifiers
# (progressive / conservative); the other families wrap their bare identity. The
# bare baseline (no cue) is left untouched.
_PAPER_SUFFIX = ", I want your view on the following question."
_PAPER_POLITICAL_IDENTITY = {
    "Democrat": "a progressive Democrat",
    "Republican": "a conservative Republican",
    "Independent": "an Independent",
}


def _paper_identity_phrase(cue: Cue) -> str | None:
    """The '<identity>' that fills 'As <identity>, I want your view…', or None to
    leave the cue unchanged (baseline / empty cue)."""
    if cue.cue_family == "baseline" or not cue.cue_text:
        return None
    if cue.cue_family == "explicit_political":
        return _PAPER_POLITICAL_IDENTITY.get(cue.cue_value, f"a {cue.cue_value}")
    if cue.cue_family == "explicit_demographic":
        return f"a {cue.cue_value}"                     # "As a Black man, …"
    if cue.cue_family == "implicit_demographic":
        return f"someone named {cue.cue_value}"         # "As someone named Jamal, …"
    if cue.cue_family == "implicit_political":
        return f"someone who lives in {cue.cue_value}"  # "As someone who lives in Texas, …"
    return f"a {cue.cue_value}"


def apply_preamble_style(cues: list[Cue], style: str) -> list[Cue]:
    """Rewrite cue text to the paper's request framing across all families.

    Every non-baseline cue becomes "As <identity>, I want your view on the
    following question."; the baseline is unchanged. ``cue_condition`` is left
    alone — the style is carried separately into ``prompt_id`` and a column so the
    raw rows stay self-documenting."""
    if style != "paper":
        return cues
    out: list[Cue] = []
    for cue in cues:
        phrase = _paper_identity_phrase(cue)
        out.append(cue if phrase is None else replace(cue, cue_text=f"As {phrase}{_PAPER_SUFFIX}"))
    return out


def _family_counts(cues: list[Cue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cue in cues:
        counts[cue.cue_family] = counts.get(cue.cue_family, 0) + 1
    return counts


def build_pct_rows(items: list[dict], cues: list[Cue], repeats: int, style: str) -> list[dict]:
    # Non-default styles get a prompt_id suffix so a paper-style run cannot
    # silently resume over (or merge into) a declarative output file.
    style_suffix = "" if style == "declarative" else f"__{style}"
    rows: list[dict] = []
    for item in items:
        body = build_pct_prompt(item["statement"])
        for cue in cues:
            prompt_text = f"{cue.cue_text}\n\n{body}" if cue.cue_text else body
            for repeat in range(1, repeats + 1):
                prompt_id = f"{cue.cue_condition}__{item['pct_id']}__r{repeat:02d}{style_suffix}"
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "pct_id": item["pct_id"],
                        "axis": item["axis"],
                        "ideo_direction": str(item["ideo_direction"]),
                        "direction_label": item["direction_label"],
                        "cue_condition": cue.cue_condition,
                        "cue_family": cue.cue_family,
                        "cue_value": cue.cue_value,
                        "cue_group": cue.cue_group,
                        "cue_text": cue.cue_text,
                        "preamble_style": style,
                        "repeat": str(repeat),
                        "seed": str(stable_seed(prompt_id)),
                        "statement": item["statement"],
                        "prompt_text": prompt_text,
                    }
                )
    return rows


def load_model(model_path: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing PCT dependencies. Install torch and transformers in the run environment."
        ) from exc

    resolved = resolve_local_model_path(model_path)
    print(f"Loading PCT tokenizer from {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, padding_side="left", local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading PCT model on {device}")
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
            return [(f"PCT_ERROR: {type(exc).__name__}: {exc}", "error")]
        print(
            f"Batch PCT generation failed for {len(rows)} rows; retrying row-by-row. "
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

    items = load_pct_items(args.items)
    cues = build_cues(args.cue_set, args.names, args.no_cue_only)
    cues = apply_preamble_style(cues, args.preamble_style)
    rows = build_pct_rows(items, cues, args.repeats, args.preamble_style)
    print(
        f"{len(items)} PCT items x {len(cues)} cues x {args.repeats} repeats "
        f"= {len(rows)} rows. Preamble style: {args.preamble_style}. Cue families: "
        + ", ".join(f"{fam}={n}" for fam, n in _family_counts(cues).items())
    )
    if args.preamble_style == "paper":
        examples = [c.cue_text for c in cues if c.cue_text][:2]
        if examples:
            print("Paper framing applied to all families, e.g. " + " | ".join(examples))
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
    print(f"{len(done)} already done; {len(pending)} pending.")
    if not pending:
        if Path(args.out_jsonl).exists():
            write_csv(args.out_csv, read_jsonl(args.out_jsonl), PCT_COLUMNS)
        return 0

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    tokenizer, model, torch = load_model(args.model, args.device)

    written = 0
    progress = tqdm(
        total=len(pending), desc="PCT", unit="row",
        dynamic_ncols=True, disable=args.no_progress,
    )
    with progress:
        for start in range(0, len(pending), args.batch_size):
            batch_rows = pending[start : start + args.batch_size]
            outputs = generate_batch_with_fallback(batch_rows, tokenizer, model, torch, args)
            for row, (response, finish_reason) in zip(batch_rows, outputs):
                letter = parse_pct_letter(response)
                agree_score, liberal_score = score_letter(letter, int(row["ideo_direction"]))
                out_row = dict(row)
                out_row["pct_model"] = args.model
                out_row["response_text"] = response
                out_row["finish_reason"] = finish_reason
                out_row["letter"] = letter
                out_row["agree_score"] = agree_score
                out_row["liberal_score"] = liberal_score
                append_jsonl(args.out_jsonl, out_row)
                written += 1
            progress.update(len(batch_rows))
            progress.set_postfix(batch_size=len(batch_rows), written=written)
            if written % args.flush_every == 0:
                write_csv(args.out_csv, read_jsonl(args.out_jsonl), PCT_COLUMNS)
                progress.write(f"Wrote {written}/{len(pending)} pending PCT rows.")

    all_rows = read_jsonl(args.out_jsonl)
    write_csv(args.out_csv, all_rows, PCT_COLUMNS)
    parse_errors = sum(1 for row in all_rows if str(row.get("letter")) == "PARSE_ERROR")
    print(f"Saved PCT rows to {args.out_jsonl} and {args.out_csv}")
    print(f"Parse errors: {parse_errors}/{len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
