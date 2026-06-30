#!/usr/bin/env python3
"""Extract residual-stream activations for the internal identity probes (arm B).

For each (cue x issue x template) we rebuild the EXACT prompt the generation run
used — the cue as an inferred user-memory in the system message, the filled
writing task as the user turn — run a single forward pass, and capture the
hidden state at the LAST token of the prompt (the position that conditions the
first generated token). Left padding puts the real last token at index -1 for
every row. We keep every layer so the probe can sweep depth.

Output (per model):
  <out-dir>/<tag>_acts.npz   float16 arrays, key "layer_{i}" -> [N, hidden]
  <out-dir>/<tag>_meta.csv   one row per N: cue_condition/family/group/value,
                             issue_id, template_id (+ instance for names/states)

Downstream: analysis/train_identity_probe.py trains the linear probes (B1),
the political-direction projection (B2), and the mediation link (B3) locally.

Open-weights only (needs hidden states) -> run on Llama/Gemma/Qwen on Brains.

    python pipeline/08_extract_activations.py --model <path> --device cuda:0 \
        --tag qwen --out-dir /data/<user>/probe_activations --templates 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import DEFAULT_GEN_MODEL, DEFAULT_ISSUES_CSV, DEFAULT_TEMPLATES_ALL_CSV, DEFAULT_WORDING_CSV
from cues import all_cues
from hf_utils import apply_chat_template, resolve_local_model_path
from io_utils import read_csv, write_csv
from prompting import (
    apply_issue_wording,
    build_system_text,
    fill_template,
    main_issues,
    slugify,
    stable_seed,
    stratified_templates,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **kwargs):  # type: ignore
        return x if x is not None else iter(())


META_COLUMNS = [
    "row_id", "cue_condition", "cue_family", "cue_group", "cue_value",
    "issue_id", "template_id", "template_rank", "seed",
]


def build_rows(n_templates: int) -> list[dict]:
    issues = apply_issue_wording(
        main_issues(read_csv(DEFAULT_ISSUES_CSV)), read_csv(DEFAULT_WORDING_CSV)
    )
    templates = stratified_templates(read_csv(DEFAULT_TEMPLATES_ALL_CSV), n_templates)
    cues = all_cues()
    rows: list[dict] = []
    for issue in issues:
        issue_id = issue.get("ces_variable", "").strip() or slugify(issue["topic_neutral"])
        topic = issue.get("prompt_topic", issue.get("topic_neutral", "")).strip()
        for template in templates:
            template_id = template.get("id", "") or f"rank_{template.get('rank')}"
            user_text = fill_template(template["selected_template"].strip(), topic)
            for cue in cues:
                row_id = f"{cue.cue_condition}__{issue_id}__{template_id}"
                rows.append({
                    "row_id": row_id,
                    "cue_condition": cue.cue_condition, "cue_family": cue.cue_family,
                    "cue_group": cue.cue_group, "cue_value": cue.cue_value,
                    "issue_id": issue_id, "template_id": template_id,
                    "template_rank": template.get("rank", ""),
                    "seed": str(stable_seed(row_id)),
                    "system_text": build_system_text(cue.cue_memory),
                    "user_text": user_text,
                })
    return rows


def load_model(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_local_model_path(model_path)
    print(f"Loading tokenizer from {resolved}")
    tok = AutoTokenizer.from_pretrained(resolved, padding_side="left", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved, torch_dtype=torch.bfloat16,
        device_map=("auto" if device == "auto" else device), local_files_only=True
    )
    model.eval()
    input_device = str(model.get_input_embeddings().weight.device) if device == "auto" else device
    return tok, model, torch, input_device


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_GEN_MODEL)
    p.add_argument("--tag", required=True, help="Short model tag for output filenames (e.g. qwen).")
    p.add_argument("--out-dir", required=True, help="Directory for the .npz/.csv (use /data on Brains).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--templates", type=int, default=5, help="Top-ranked templates to cross (task variation).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.templates)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Built {len(rows)} (cue x issue x template) prompts "
          f"[{len({r['cue_condition'] for r in rows})} cues].")

    tok, model, torch, input_device = load_model(args.model, args.device)

    layer_chunks: dict[int, list] = {}
    n_layers = None
    bar = tqdm(total=len(rows), desc=f"acts:{args.tag}", unit="row")
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        formatted = [apply_chat_template(tok, r["user_text"], r["system_text"]) for r in batch]
        inputs = tok(
            formatted, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_input_tokens,
        ).to(input_device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple (n_layers+1) of [B, T, H]
        if n_layers is None:
            n_layers = len(hs)
            layer_chunks = {i: [] for i in range(n_layers)}
        for i in range(n_layers):
            # float32 not float16: Gemma-3 residual-stream outliers exceed fp16 max
            # (65504) and would become inf.
            last = hs[i][:, -1, :].float().cpu().numpy()  # left-padded -> -1 is real
            layer_chunks[i].append(last)
        if hasattr(bar, "update"):
            bar.update(len(batch))

    acts = {f"layer_{i}": np.concatenate(layer_chunks[i], axis=0) for i in range(n_layers)}
    npz_path = out_dir / f"{args.tag}_acts.npz"
    np.savez(npz_path, **acts)
    write_csv(out_dir / f"{args.tag}_meta.csv", rows, META_COLUMNS)
    shape = acts["layer_0"].shape
    print(f"Saved {n_layers} layers x {shape[0]} rows x {shape[1]} dims -> {npz_path}")
    print(f"Saved metadata -> {out_dir / f'{args.tag}_meta.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
