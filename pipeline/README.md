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
# 1. Materialise the first-name list, reproduced verbatim from Tonneau et al.
#    Appendix A.1 (Rosenman, Elder-Hayes, Tzioumis; Black/White x man/woman; 50
#    names per source x cell). Writes the committed data/input/names/names.csv
#    (600 rows). No downloads or derivation.
python pipeline/build_name_list.py

# 2. Probe the GENERATION model cue-only: per name, three separate prompts infer
#    race {Black,White,Unknown}, gender {man,woman,Unknown} (Tonneau's verbatim
#    forced-choice annotation prompt), and a continuous political lean in [-1,+1]
#    (same scale as liberal_score). 3 repeats.
python pipeline/05_run_cue_probe.py \
  --names data/input/names/names.csv \
  --out-jsonl "$OUT/cue_probe.jsonl" --out-csv "$OUT/cue_probe.csv" \
  --device cuda:0 --batch-size 96 --repeats 3

# 3. Summarise: race/gender recall + 'Cannot tell' abstention per subgroup, and
#    inferred political lean (name-clustered bootstrap CI) by subgroup.
python analysis/cue_probe_report.py --probe "$OUT/cue_probe.csv" \
  --out-dir results/cue_probe --figures-dir figures/cue_probe
```

Three separate prompts per name (not one joint prompt) so a race/gender guess
cannot prime the political-lean guess. `05_run_cue_probe.py` honours the same
`--num-shards`/`--shard-index`, resume, and seeding conventions as `02`/`03`.

Cue-only (above) is the primary measure. The **ecological** variant places each
name inside a real writing request (the rank-1 template filled with each of the
19 main issues) so legibility is measured under generation conditions — "does the
conclusion survive when the name sits inside a task?". It multiplies rows by the
issue count, so run it at `--repeats 1` (600 names x 19 issues x 3 attributes =
34,200 probes, ~20 min on one GPU):

```bash
python pipeline/05_run_cue_probe.py --names data/input/names/names.csv \
  --questions --repeats 1 \
  --out-jsonl "$OUT/cue_probe_questions.jsonl" --out-csv "$OUT/cue_probe_questions.csv" \
  --device cuda:0 --batch-size 96
```

## Political Compass Test (standalone)

A second arm that puts the **same name cues** in front of a fixed survey
instrument instead of an open writing task, adapting Rozado / cssmodels
(`biasissycophancy`). It answers a complementary question to the main run: does
the implicit demographic cue move the model's *self-reported* political position?

The 62-item instrument and its left/right coding are committed verbatim at
`data/input/pct/pct_items_coded.csv` (`axis` = economic/social, `ideo_direction`
= -1 left / +1 right / 0 ambiguous). Each item is asked as a forced-choice
Likert block (A Strongly Agree … D Strongly Disagree). Two arms run over it:

- **baseline** — the bare Likert prompt, no cue.
- **implicit-demographic cue** — each item prepended with `My name is X.`, the
  same 12 generation names (`data/input/names/names_generation.csv`) and the
  byte-identical cue string used by the main run.

Letters are scored to the pipeline's `liberal_score` axis (+1 liberal/left, -1
conservative/right): `agree_score x (-ideo_direction) / 2`. Ambiguous items get
no `liberal_score`.

```text
62 PCT items
13 cues (1 no-cue baseline + 12 name cues)
 3 stochastic repeats
= 2,418 rows
```

```bash
# Run baseline + name cues on the GENERATION model (same loader/seeding as 02).
python pipeline/06_run_pct.py \
  --out-jsonl "$OUT/pct.jsonl" --out-csv "$OUT/pct.csv" \
  --device cuda:0 --batch-size 16 --repeats 3

# Summarise: PCT lean by condition (overall + economic/social axes), the
# name-cue effect vs baseline (paired within item, item-clustered bootstrap),
# and a 2-D compass scatter.
python analysis/pct_report.py --pct "$OUT/pct.csv" \
  --out-dir results/pct --figures-dir figures/pct
```

`--cue-set all` runs the full 29-cue grid instead of baseline + names: explicit
political (Democrat / Republican / Independent), implicit political (red/blue/
swing states), explicit demographic ("I am a White man."), and the implicit
demographic names. The explicit-political cues are the manipulation check —
following Tornberg & Schimmel (arXiv:2604.27633), an explicit conservative cue
should swing the PCT sharply right while the progressive cue barely moves it
(the baseline already sits near the left ceiling), which is what makes a
near-zero implicit-name effect interpretable. The report keys conditions by
`cue_family`/`cue_group` so explicit and implicit demographic cues (which share
a cue_group) stay distinct.

```text
62 PCT items x 29 cues x 3 repeats = 5,394 rows
```

```bash
python pipeline/06_run_pct.py --cue-set all \
  --out-jsonl "$OUT/pct_all.jsonl" --out-csv "$OUT/pct_all.csv" \
  --device cuda:0 --batch-size 16 --repeats 3
python analysis/pct_report.py --pct "$OUT/pct_all.csv" \
  --out-dir results/pct_all --figures-dir figures/pct_all
```

By default the explicit-political cues are bare self-identifications
(`I am a Republican.`), consistent with every other arm of the pipeline. To get
closer to the paper's effect magnitude, `--preamble-style paper` rewrites *only*
the explicit-political cues to Tornberg & Schimmel's verbatim request framing
(`As a conservative Republican, I want your view on the following question.`);
all demographic, name, state, and baseline cues stay declarative. The style is
recorded in a `preamble_style` column and suffixed into `prompt_id`, so a
paper-style run cannot resume over or merge into a declarative output — **write
it to a separate file**:

```bash
python pipeline/06_run_pct.py --cue-set all --preamble-style paper \
  --out-jsonl "$OUT/pct_all_paper.jsonl" --out-csv "$OUT/pct_all_paper.csv" \
  --device cuda:0 --batch-size 16 --repeats 3
python analysis/pct_report.py --pct "$OUT/pct_all_paper.csv" \
  --out-dir results/pct_all_paper --figures-dir figures/pct_all_paper
```

`06_run_pct.py` honours the same `--num-shards`/`--shard-index`, resume,
`--overwrite`, and seeding conventions as `02`/`05`. Use `--no-cue-only` to run
the baseline arm alone. The 2-D scatter uses our transparent per-axis
`liberal_score` means, not politicalcompass.org's proprietary coordinates.

## Tests

```bash
python -m unittest discover -s pipeline/tests -p "test_*.py"
```
