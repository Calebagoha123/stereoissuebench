# Thesis Framing Pipeline

This directory contains the model-side pipeline used to create
`data/processed/evaluated_with_effects.csv`.

It is intentionally separate from the report plotting script in
`scripts/make_report_figures.py`:

- this pipeline builds prompts, runs generation, runs stance judging, and adds
  matched no-cue baselines;
- the top-level report script joins the scored model rows to CES microdata and
  creates the final figures.

## Design

The pilot design is:

```text
17 main CES-linked issues
30 stratified writing templates per issue
29 cue realizations
3 stochastic repeats
= 44,370 prompt rows
```

`issue_prompt_wording.csv` supplies open-direction CES-style policy phrases for
the prompt topic. `input_data/issues_experiment.csv` supplies the CES variables,
stance labels, and liberal-sign recoding metadata. `input_data/templates_selected.csv`
supplies the writing-assistance templates.

## Commands

From the repository root:

```bash
python thesis_framing_pipeline/00_validate_inputs.py
python thesis_framing_pipeline/01_build_prompts.py --mode pilot
```

Run generation:

```bash
python thesis_framing_pipeline/02_run_generation.py \
  --prompts thesis_framing_pipeline/results/prompts_pilot.csv \
  --out-jsonl thesis_framing_pipeline/results/generations_pilot.jsonl \
  --out-csv thesis_framing_pipeline/results/generations_pilot.csv \
  --device cuda:0 \
  --batch-size 8 \
  --max-new-tokens 1000
```

Run stance evaluation:

```bash
python thesis_framing_pipeline/03_run_stance_eval.py \
  --generations thesis_framing_pipeline/results/generations_pilot.csv \
  --out-jsonl thesis_framing_pipeline/results/evaluated_pilot.jsonl \
  --out-csv thesis_framing_pipeline/results/evaluated_pilot.csv \
  --device cuda:0 \
  --batch-size 16
```

Add matched no-cue baselines and summary CSVs:

```bash
python thesis_framing_pipeline/04_analyse_prelim.py \
  --evaluated thesis_framing_pipeline/results/evaluated_pilot.csv \
  --out-dir thesis_framing_pipeline/results/analysis_pilot
```

The report figures are then generated from the resulting
`evaluated_with_effects.csv` with:

```bash
python scripts/make_report_figures.py --ces-dta ../CES/CES25_Common.dta
```

## Model Defaults

Defaults are set in `config.py`:

- generation model: `/data/resource/huggingface/models--Qwen--Qwen3.5-9B`
- stance judge: `/data/resource/huggingface/models--Qwen--Qwen3.5-4B`

The loaders resolve either direct snapshot directories or Hugging Face
`models--.../snapshots/...` cache directories and use local files only.

## Sharding

`02_run_generation.py` and `03_run_stance_eval.py` support deterministic row
sharding with `--num-shards` and `--shard-index`. Use separate output files per
shard, then merge them with `merge_generation_outputs.py` or
`merge_eval_outputs.py`.
