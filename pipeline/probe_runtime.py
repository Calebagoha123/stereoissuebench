"""Shared local-HF batched generation for the probe runners.

Factored out of 05_run_cue_probe.py so the belief/relevance runner (07) can reuse
the exact same model loading, seeding, batched sampling, and row-by-row error
fallback without importing a numerically-named module. 05 keeps its own copy to
stay self-contained; this is the canonical version for new runners.
"""

from __future__ import annotations

import hashlib
import random
import sys

from hf_utils import apply_chat_template, resolve_local_model_path


def load_model(model_path: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing probe dependencies. Install torch and transformers in the run environment."
        ) from exc

    resolved = resolve_local_model_path(model_path)
    print(f"Loading tokenizer from {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(resolved, padding_side="left", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved, torch_dtype=torch.bfloat16,
        device_map=("auto" if device == "auto" else device), local_files_only=True
    )
    model.eval()
    # With device_map="auto" the model shards across visible GPUs; inputs go to
    # the embedding's device. Otherwise inputs go to the single chosen device.
    input_device = str(model.get_input_embeddings().weight.device) if device == "auto" else device
    return tokenizer, model, torch, input_device


def seed_for_batch(rows: list[dict], torch) -> None:
    seed_source = "|".join(f"{row['prompt_id']}:{row['seed']}" for row in rows)
    seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_batch(rows: list[dict], tokenizer, model, torch, args) -> list[str]:
    """Sample a continuation per row from its ``prompt_text`` (single user turn)."""
    seed_for_batch(rows, torch)
    formatted = [apply_chat_template(tokenizer, row["prompt_text"]) for row in rows]
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_input_tokens,
    ).to(args.device)
    do_sample = args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
        for ids in output_ids
    ]


def generate_batch_with_fallback(rows, tokenizer, model, torch, args) -> list[tuple[str, str]]:
    try:
        return [(text, "ok") for text in generate_batch(rows, tokenizer, model, torch, args)]
    except Exception as exc:  # noqa: BLE001
        if len(rows) == 1:
            return [(f"PROBE_ERROR: {type(exc).__name__}: {exc}", "error")]
        print(
            f"Batch failed for {len(rows)} rows; retrying row-by-row. "
            f"Error was {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        outputs: list[tuple[str, str]] = []
        for row in rows:
            outputs.extend(generate_batch_with_fallback([row], tokenizer, model, torch, args))
        return outputs
