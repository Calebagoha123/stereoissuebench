# Pipeline

The model-side pipeline: it builds matched prompt rows, runs generation, judges
stance, and adds matched no-cue baselines. The report figures are produced
separately by `analysis/make_report_figures.py` from the scored rows.

## Inputs and Outputs

Committed inputs are read directly from the repo (`pipeline/config.py`):

- `data/input/issues_experiment.csv` — CES variables, stance labels, and
  liberal-sign recoding metadata.
- `data/input/issue_prompt_wording.csv` — open-direction CES-style policy
  phrasing keyed by `ces_variable`.
- `data/input/templates_run_30.csv` — the 30 most stance-eliciting writing
  templates (the run target), ranked from `data/input/templates_pool_145.csv`.

Large generated artifacts (prompts, generations, scored rows) are written to the
**output root**, resolved as:

1. `$THESIS_PIPELINE_DATA_ROOT` if set, else
2. the VM scratch volume `/data/kell8360/thesis_framing_pipeline` if present, else
3. the gitignored `data/processed/` in the repo.

## Run Grid

```text
19 main CES-linked issues (analysis_tier == main)
30 stance-ranked writing templates per issue
29 cue realizations (incl. no-cue baseline)
 3 stochastic repeats
= 49,590 prompt rows
```

## One-Shot Run

```bash
bash pipeline/prepare_data_root.sh     # create the output results dir
bash pipeline/run_pipeline.sh pilot    # validate → build → generate → score → analyse
```

Use `smoke` instead of `pilot` for a 1-issue / 2-template / 3-cue dry run.

## Per-Step Commands

`$OUT` below is the output root's `results/` directory.

```bash
python pipeline/00_validate_inputs.py
python pipeline/01_build_prompts.py --mode pilot

python pipeline/02_run_generation.py \
  --prompts "$OUT/prompts_pilot.csv" \
  --out-jsonl "$OUT/generations_pilot.jsonl" \
  --out-csv "$OUT/generations_pilot.csv" \
  --device cuda:0 --batch-size 8 --max-new-tokens 1000

python pipeline/03_run_stance_eval.py \
  --generations "$OUT/generations_pilot.csv" \
  --out-jsonl "$OUT/evaluated_pilot.jsonl" \
  --out-csv "$OUT/evaluated_pilot.csv" \
  --device cuda:0 --batch-size 16

python pipeline/04_analyse_prelim.py \
  --evaluated "$OUT/evaluated_pilot.csv" \
  --out-dir "$OUT/analysis_pilot"
```

`04_analyse_prelim.py` writes `evaluated_with_effects.csv` (the scored rows with
matched baselines) into its `--out-dir`. Copy that file into `data/processed/`
to feed `analysis/make_report_figures.py`.

## Model Defaults

Set in `config.py`:

- generation model: `/data/resource/huggingface/models--Qwen--Qwen3.5-9B`
- stance judge: `/data/resource/huggingface/models--Qwen--Qwen3.5-4B`

The loaders resolve either direct snapshot directories or Hugging Face
`models--.../snapshots/...` cache directories and use local files only.

## Sharding

`02_run_generation.py` and `03_run_stance_eval.py` support deterministic row
sharding with `--num-shards` and `--shard-index`. Use separate output files per
shard, then merge with `merge_generation_outputs.py` / `merge_eval_outputs.py`.

## Cue-Legibility Probe (standalone)

A separate manipulation check, following Tonneau et al. (arXiv:2601.18486): does
a first-name cue carry enough signal for the model to infer the user's profile?
It is independent of the main stance run and uses its own name set.

```bash
# 1. Build the race x gender name list (Tonneau Appendix A.1 recipe:
#    Tzioumis race-specificity x SSA gender shares, race-specific freq floors).
#    Writes the committed data/input/names/names_tzioumis.csv (200 names, 50/cell).
python pipeline/build_name_list.py --source tzioumis --per-cell 50

# 2. Probe the GENERATION model cue-only: per name, three separate prompts infer
#    race {White,Black,Other,Cannot tell}, gender {man,woman,Cannot tell}, and a
#    continuous political lean in [-1,+1] (same scale as liberal_score). 3 repeats.
python pipeline/05_run_cue_probe.py \
  --names data/input/names/names_tzioumis.csv \
  --out-jsonl "$OUT/cue_probe.jsonl" --out-csv "$OUT/cue_probe.csv" \
  --device cuda:0 --batch-size 16 --repeats 3

# 3. Summarise: race/gender recall + 'Cannot tell' abstention per subgroup, and
#    inferred political lean (name-clustered bootstrap CI) by subgroup.
python analysis/cue_probe_report.py --probe "$OUT/cue_probe.csv" \
  --out-dir results/cue_probe --figures-dir figures/cue_probe
```

Three separate prompts per name (not one joint prompt) so a race/gender guess
cannot prime the political-lean guess. `05_run_cue_probe.py` honours the same
`--num-shards`/`--shard-index`, resume, and seeding conventions as `02`/`03`.

## Tests

```bash
python -m unittest discover -s pipeline/tests -p "test_*.py"
```
