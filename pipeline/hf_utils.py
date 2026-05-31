"""Hugging Face model-loading helpers used by generation and stance eval."""

from __future__ import annotations

from pathlib import Path


def resolve_local_model_path(model_or_path: str) -> str:
    """Resolve either a direct snapshot path or a HF cache models-- directory."""

    path = Path(model_or_path)
    if not path.exists():
        return model_or_path
    if (path / "config.json").exists():
        return str(path)
    snapshots = path / "snapshots"
    if snapshots.exists():
        candidates = [p for p in snapshots.iterdir() if (p / "config.json").exists()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(candidates[0])
    return str(path)


def apply_chat_template(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

