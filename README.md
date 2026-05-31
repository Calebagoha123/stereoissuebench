# Identity-Cued Political Stance in Writing Assistance

A reproducible research artifact for a focused experiment: does a writing-
assistance model change its political stance when the prompt contains a user
identity cue? The report compares model cue effects against subgroup opinion in
CES 2025 over CES-linked, IssueBench-style policy items.

## Repository Layout

```text
.
├── data/
│   ├── input/                       # committed experiment inputs
│   │   ├── issues_experiment.csv        # CES-linked policy issues (19 main + sensitivity/robustness)
│   │   ├── issue_prompt_wording.csv     # open-direction CES-style prompt phrasing per issue
│   │   ├── templates_pool_145.csv       # the 145-template candidate pool
│   │   └── templates_run_30.csv         # the 30 most stance-eliciting templates (run target)
│   ├── reference/
│   │   └── ces_ground_truth_template.csv   # CES issue recoding metadata
│   └── processed/                   # large generated outputs (gitignored)
├── pipeline/                        # model side: build prompts → generate → score stance
├── analysis/                        # report side: figures + estimate tables from scored rows
│   ├── make_report_figures.py
│   └── recompute_nonneutral_baseline.py
├── results/
│   ├── main/                        # primary estimate tables
│   ├── nonneutral/                  # non-neutral-baseline variant tables
│   └── ces_directionality_validation.csv
└── figures/
    ├── main/                        # primary report figures
    └── nonneutral/                  # non-neutral-baseline variant figures
```

## Experimental Design

The run grid is:

- **19** CES-linked main policy issues (`analysis_tier == main`; the issues file
  also carries a few `sensitivity`/`robustness` items that are not used in the
  main run)
- **30** writing templates per issue — the most stance-eliciting templates in
  `data/input/templates_run_30.csv`, ranked from the 145-template pool
- **29** cue realizations, including a no-cue baseline
- **3** stochastic repeats

= **49,590** prompt rows.

Model output is scored on a liberal-score scale from `-1` (more conservative) to
`+1` (more liberal). CES items are recoded to the same scale using
`data/reference/ces_ground_truth_template.csv`.

Cue groups:

- Explicit political: Republican, Independent, Democrat
- Explicit demographic: White man, White woman, Black man, Black woman
- Implicit political: red, swing, and blue state residence
- Implicit demographic: first-name cues grouped by race and gender

## What Is Reported

Primary figures in `figures/main/`:

- `model_vs_ces_levels.png` — model mean liberal-score and CES weighted subgroup
  mean by cue group.
- `model_cue_effects.png` — model cue effect relative to the no-cue baseline.
- `model_vs_ces_did.png` — difference-in-differences: model `(cued - baseline)`
  against CES `(subgroup - population)`.

The tables behind them are `results/main/cue_ces_estimates.csv` and
`results/main/cue_ces_by_issue.csv`. The `nonneutral/` directories hold the same
artifacts computed against a baseline restricted to non-neutral baseline rows
(see `analysis/recompute_nonneutral_baseline.py`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce The Report Figures

The figure script reads the scored model rows, the CES recoding metadata, and
respondent-level CES microdata (passed with `--ces-dta`; not committed):

```bash
python analysis/make_report_figures.py \
  --evaluated data/processed/evaluated_with_effects.csv \
  --ces-dta ../CES/CES25_Common.dta \
  --bootstrap 1000
```

This writes `figures/main/*.png`, `results/main/cue_ces_estimates.csv`, and
`results/main/cue_ces_by_issue.csv`. For the non-neutral-baseline variant, first
recompute the baseline, then point the figure script at the output dirs:

```bash
python analysis/recompute_nonneutral_baseline.py
python analysis/make_report_figures.py \
  --evaluated data/processed/evaluated_nonneutral_recomputed.csv \
  --ces-dta ../CES/CES25_Common.dta \
  --figures-dir figures/nonneutral \
  --results-dir results/nonneutral
```

## Reproduce Model Scoring (GPU / VM)

`pipeline/` builds prompts, runs generation, judges stance, and adds matched
no-cue baselines. Committed inputs are read directly from `data/input/`; large
generated outputs go to an output root (`$THESIS_PIPELINE_DATA_ROOT`, else a VM
scratch volume, else the gitignored `data/processed/`).

On the VM, after pulling the repo:

```bash
export THESIS_PIPELINE_DATA_ROOT=/data/<user>/thesis_framing_pipeline   # optional
bash pipeline/prepare_data_root.sh        # creates the output results dir
bash pipeline/run_pipeline.sh pilot       # full build → generate → score → analyse
```

`run_pipeline.sh smoke` runs a tiny 1-issue / 2-template slice for a quick check.
Generation and judging honor `GEN_DEVICE`, `EVAL_DEVICE`, `GEN_BATCH_SIZE`,
`EVAL_BATCH_SIZE`, and `GEN_MAX_NEW_TOKENS`. The scored output
`evaluated_with_effects.csv` (and the `analysis_pilot/` summaries) land under the
output root; copy `evaluated_with_effects.csv` into `data/processed/` to feed the
report figures. See `pipeline/README.md` for the per-step commands, model
defaults, and sharding.

## Notes

- Survey weights use CES `commonweight`.
- CES state cue groups use CA/MA/NY (blue), AL/OK/TX (red), GA/PA/WI (swing),
  matching the prompt cues.
- Binary support/oppose items map support to `+1` and oppose to `-1` before
  applying each issue's `liberal_sign`; the abortion item is ordinal and mapped
  from CES codes `1..4` onto `[-1, +1]`.
