#!/usr/bin/env python3
"""Causal test (arm B, steering): push the internal Dem-Rep axis and watch stance.

The probes showed the internal political axis *predicts* written stance (mediation
r~0.8-0.9). This tests whether that axis is *causal*: we add c * sep_H * v_hat_H
to the residual stream at a few mid layers during generation (v_hat = the unit
Democrat-minus-Republican direction at hidden index H, sep = the natural Dem-Rep
gap on that axis, both precomputed by analysis from the saved activations), sweep
the coefficient c, and re-score the generations with the DeBERTa stance scorer.
If mean liberal-score rises monotonically with c, the axis is a real lever.

Steered prompts are the BASELINE (no-cue) writing tasks, so the only thing moving
the output is our intervention. Output feeds stance_model/predict.py for scoring.

    python pipeline/09_steer_generate.py --model <llama> --device cuda:0 \
        --steer-npz <dir>/llama_steer_dirs.npz --hidden-indices 12,16,20 \
        --coeffs -4,-2,-1,0,1,2,4 --templates 2 \
        --out-jsonl <out>/steer_llama.jsonl --out-csv <out>/steer_llama.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import DEFAULT_GEN_MODEL, DEFAULT_ISSUES_CSV, DEFAULT_RESULTS_DIR, DEFAULT_TEMPLATES_ALL_CSV, DEFAULT_WORDING_CSV
from hf_utils import apply_chat_template, resolve_local_model_path
from io_utils import append_jsonl, read_csv, read_jsonl, write_csv
from prompting import apply_issue_wording, fill_template, main_issues, slugify, stable_seed, stratified_templates

OUT_COLUMNS = [
    "prompt_id", "issue_id", "ces_variable", "stance_target", "liberal_sign",
    "template_id", "steer_coeff", "seed", "prompt_text",
    "steer_model", "response_text", "finish_reason",
]


def build_prompts(n_templates: int) -> list[dict]:
    issues = apply_issue_wording(main_issues(read_csv(DEFAULT_ISSUES_CSV)), read_csv(DEFAULT_WORDING_CSV))
    templates = stratified_templates(read_csv(DEFAULT_TEMPLATES_ALL_CSV), n_templates)
    rows = []
    for issue in issues:
        issue_id = issue.get("ces_variable", "").strip() or slugify(issue["topic_neutral"])
        topic = issue.get("prompt_topic", issue.get("topic_neutral", "")).strip()
        for t in templates:
            tid = t.get("id", "") or f"rank_{t.get('rank')}"
            rows.append({
                "prompt_id": f"{issue_id}__{tid}", "issue_id": issue_id,
                "ces_variable": issue.get("ces_variable", ""),
                "stance_target": issue.get("stance_target", ""),
                "liberal_sign": issue.get("liberal_sign", ""),
                "template_id": tid,
                "prompt_text": fill_template(t["selected_template"].strip(), topic),
            })
    return rows


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_GEN_MODEL)
    p.add_argument("--steer-npz", required=True)
    p.add_argument("--hidden-indices", default="12,16,20", help="hidden_states indices to steer (H>=1)")
    p.add_argument("--coeffs", default="-4,-2,-1,0,1,2,4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--templates", type=int, default=2)
    p.add_argument("--out-jsonl", default=str(DEFAULT_RESULTS_DIR / "steer.jsonl"))
    p.add_argument("--out-csv", default=str(DEFAULT_RESULTS_DIR / "steer.csv"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--max-input-tokens", type=int, default=512)
    args = p.parse_args()

    indices = [int(x) for x in args.hidden_indices.split(",") if x.strip()]
    coeffs = [float(x) for x in args.coeffs.split(",") if x.strip()]
    prompts = build_prompts(args.templates)
    print(f"{len(prompts)} base prompts x {len(coeffs)} coeffs, steering hidden indices {indices}")

    resolved = resolve_local_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(resolved, padding_side="left", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved, torch_dtype=torch.bfloat16, device_map=args.device, local_files_only=True).eval()

    sd = np.load(args.steer_npz)
    state = {"c": 0.0}
    vecs, seps, handles = {}, {}, []
    for H in indices:
        vh = torch.tensor(sd[f"dir_{H}"], dtype=model.dtype, device=args.device)
        vecs[H] = vh
        seps[H] = float(sd[f"sep_{H}"])

    def make_hook(H):
        def hook(module, inp, out):
            if state["c"] == 0.0:
                return out
            add = (state["c"] * seps[H]) * vecs[H]
            if isinstance(out, tuple):
                return (out[0] + add,) + tuple(out[1:])
            return out + add
        return hook

    layers = model.model.layers
    for H in indices:  # hidden_states[H] is the output of decoder block H-1
        handles.append(layers[H - 1].register_forward_hook(make_hook(H)))

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out_jsonl).exists():
        Path(args.out_jsonl).unlink()
    written = 0
    for c in coeffs:
        state["c"] = c
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            formatted = [apply_chat_template(tok, r["prompt_text"]) for r in batch]
            enc = tok(formatted, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_input_tokens).to(args.device)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            ilen = enc["input_ids"].shape[1]
            for r, seq in zip(batch, ids):
                text = tok.decode(seq[ilen:], skip_special_tokens=True).strip()
                out = dict(r)
                out.update(prompt_id=f"{r['prompt_id']}__c{c:+g}", steer_coeff=c,
                           seed=str(stable_seed(f"{r['prompt_id']}{c}")),
                           steer_model=args.model, response_text=text, finish_reason="ok")
                append_jsonl(args.out_jsonl, out)
                written += 1
        print(f"  coeff {c:+g}: {written} total generations", flush=True)

    write_csv(args.out_csv, read_jsonl(args.out_jsonl), OUT_COLUMNS)
    print(f"Saved {written} steered generations -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
