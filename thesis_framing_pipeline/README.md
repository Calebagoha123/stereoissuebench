# Thesis Framing Pipeline

Purpose-specific preliminary-results pipeline for the Qwen-only thesis run.

## Inputs

Defaults point to the packaged thesis CSVs:

- local fallback: `thesis_framing_pipeline/input_data/`
- VM default when present: `/data/kell8360/thesis_framing_pipeline/input_data/`

Runtime results default to `/data/kell8360/thesis_framing_pipeline/results/`
when that directory exists. Override with `THESIS_PIPELINE_DATA_ROOT`.

On the VM, prepare the data root after pulling:

```bash
cd /home/kell8360/stereoissuebench
bash thesis_framing_pipeline/prepare_data_root.sh
```

The pilot uses `analysis_tier == "main"`, a stratified 30-template subset, 29 cue
realizations, and 3 repeats:

`17 issues * 30 templates * 29 cues * 3 repeats = 44,370 prompt rows`

`issue_prompt_wording.csv` supplies the exact open-direction CES-style policy
phrases used to fill template `X`. The original `topic_support` and
`topic_oppose` columns remain the stance-judge labels.

## Quick Commands

From the repository root:

```bash
bash stereoissuebench/thesis_framing_pipeline/run_prelim.sh smoke
```

Build the full pilot prompt file only:

```bash
python3 stereoissuebench/thesis_framing_pipeline/00_validate_inputs.py
python3 stereoissuebench/thesis_framing_pipeline/01_build_prompts.py --mode pilot
```

Run generation and stance evaluation:

```bash
python3 stereoissuebench/thesis_framing_pipeline/02_run_generation.py \
  --prompts stereoissuebench/thesis_framing_pipeline/results/prompts_pilot.csv \
  --out-jsonl stereoissuebench/thesis_framing_pipeline/results/generations_pilot.jsonl \
  --out-csv stereoissuebench/thesis_framing_pipeline/results/generations_pilot.csv \
  --device cuda:0 \
  --batch-size 8

python3 stereoissuebench/thesis_framing_pipeline/03_run_stance_eval.py \
  --generations stereoissuebench/thesis_framing_pipeline/results/generations_pilot.csv \
  --out-jsonl stereoissuebench/thesis_framing_pipeline/results/evaluated_pilot.jsonl \
  --out-csv stereoissuebench/thesis_framing_pipeline/results/evaluated_pilot.csv \
  --device cuda:0 \
  --batch-size 16
```

Both model scripts resume from existing JSONL outputs by `prompt_id`.
The shell runner also accepts environment overrides:

```bash
GEN_DEVICE=cuda:2 EVAL_DEVICE=cuda:2 GEN_BATCH_SIZE=4 EVAL_BATCH_SIZE=8 \
  bash stereoissuebench/thesis_framing_pipeline/run_prelim.sh pilot
```

Batched stochastic generation is reproducible for the same prompt order and
batch size. Use `--strict-row-seeds` if you need one-row-at-a-time sampling with
each row's stored seed applied independently.

Run preliminary summaries and figures:

```bash
python3 stereoissuebench/thesis_framing_pipeline/04_analyse_prelim.py \
  --evaluated stereoissuebench/thesis_framing_pipeline/results/evaluated_pilot.csv \
  --out-dir stereoissuebench/thesis_framing_pipeline/results/analysis_pilot
```

## Model Defaults

- Generation: `/data/resource/huggingface/models--Qwen--Qwen3.5-9B`
- Stance judge: `/data/resource/huggingface/models--Qwen--Qwen3.5-4B`

The loaders resolve either direct snapshot directories or Hugging Face
`models--.../snapshots/...` cache directories and use `local_files_only=True`.
