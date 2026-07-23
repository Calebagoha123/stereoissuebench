#!/usr/bin/env python3
"""vLLM version of the 0-100 codebook stance judge (stance_model/judge_score100.py).

Same prompt / anchoring / parsing as the transformers judge, but uses vLLM's
continuous batching so it scales to corpus-size judging (thousands+) in minutes.
For the 150-item validation set the two are ~equivalent (load-dominated); use this
for the corpus sample and any larger pass.

Input needs `stance_target` (proposition) + `response_text`. Output adds
`judge_score` (0-100 float or blank) and `judge_raw`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from judge_score100 import PROMPT, parse_score, proposition_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    path = Path(args.generations)
    df = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
    if args.limit:
        df = df.head(args.limit).copy()
    props = [proposition_for(r) for _, r in df.iterrows()]
    texts = df["response_text"].astype(str).tolist()

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, dtype="bfloat16")
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(proposition=p, response_text=t)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p, t in zip(props, texts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outs = llm.generate(prompts, sp)
    raws = [o.outputs[0].text.strip() for o in outs]

    df["judge_raw"] = raws
    df["judge_score"] = [parse_score(r) for r in raws]
    df["judge_model"] = args.model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_bad = df["judge_score"].isna().sum()
    print(f"Judged {len(df)} -> {out}  (unratable/parse-fail: {n_bad})")
    print(df["judge_score"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
