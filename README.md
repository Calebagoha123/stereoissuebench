# Identity-Cued Political Stance in Writing Assistance

This repository is a reproducible research artifact for a focused experiment:
whether a writing-assistance model changes its political stance when the prompt
contains a user identity cue.

The current report uses Qwen generations over CES-linked IssueBench-style policy
items, then compares model cue effects against subgroup opinion in CES 2025.

## What Is Reported

The retained report artifacts are:

- [figures/model_vs_ces_levels.png](figures/model_vs_ces_levels.png)  
  Model mean liberal-score and CES weighted subgroup mean by cue group.
- [figures/model_cue_effects.png](figures/model_cue_effects.png)  
  Model cue effect relative to the no-cue baseline.
- [figures/model_vs_ces_did.png](figures/model_vs_ces_did.png)  
  Difference-in-differences comparison: model `(cued - baseline)` against CES
  `(subgroup - population)`.

The numeric tables behind the figures are:

- [results/cue_ces_estimates.csv](results/cue_ces_estimates.csv)
- [results/cue_ces_by_issue.csv](results/cue_ces_by_issue.csv)

## Experimental Design

The scored model data contains 44,370 generations, of which 44,315 received a
valid stance score. The design is:

- 17 CES-linked policy issues
- 30 writing templates per issue
- 29 cue realizations, including no-cue baseline
- 3 stochastic repeats

The model output is scored on a liberal-score scale from `-1` to `+1`, where
negative means more conservative and positive means more liberal. The CES items
are recoded to the same scale using `data/reference/ces_ground_truth_template.csv`.

Cue groups:

- Explicit political: Republican, Independent, Democrat
- Explicit demographic: White man, White woman, Black man, Black woman
- Implicit political: red, swing, and blue state residence
- Implicit demographic: first-name cues grouped by race and gender

## Key Empirical Readout

The model has a liberal no-cue baseline relative to the CES population mean
(`0.395` vs `0.086` averaged across the 17 policy items). The difference-in-
differences plot therefore asks a narrower question: whether cues move the model
by the same amount that the corresponding subgroup differs from the population.

Main patterns in `results/cue_ces_estimates.csv`:

- Republican cue overshoots the CES subgroup shift: model `-0.787`, CES `-0.437`.
- Democrat cue undershoots the CES subgroup shift: model `+0.301`, CES `+0.397`.
- Explicit Black demographic cues overshoot CES race-gender shifts.
- State cues are comparatively well calibrated, especially blue and swing states.
- Name cues are compressed and partly wrong-signed; white-male names shift the
  model slightly liberal even though the CES subgroup shift is conservative.

## Repository Layout

```text
.
├── data/
│   ├── processed/                                # local/generated scored model rows
│   └── reference/ces_ground_truth_template.csv   # CES issue recoding metadata
├── figures/                                      # three report figures
├── results/                                      # aggregate estimate tables
├── scripts/make_report_figures.py                # reproduces figures/results
└── thesis_framing_pipeline/                      # prompt, generation, judging, and prelim analysis pipeline
```

The scored model rows and CES microdata are not committed. The figure script
expects the scored rows at `data/processed/evaluated_with_effects.csv` and CES
at `../CES/CES25_Common.dta` by default; both paths can be overridden.

## Reproduce The Report Figures

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the three figures and result tables:

```bash
python scripts/make_report_figures.py \
  --ces-dta ../CES/CES25_Common.dta \
  --bootstrap 1000
```

This reads local/generated inputs:

- `data/processed/evaluated_with_effects.csv`
- `data/reference/ces_ground_truth_template.csv`
- respondent-level CES microdata passed with `--ces-dta`

and writes:

- `figures/model_vs_ces_levels.png`
- `figures/model_cue_effects.png`
- `figures/model_vs_ces_did.png`
- `results/cue_ces_estimates.csv`
- `results/cue_ces_by_issue.csv`

For a fast smoke check, use a smaller bootstrap count:

```bash
python scripts/make_report_figures.py \
  --ces-dta ../CES/CES25_Common.dta \
  --bootstrap 20 \
  --figures-dir /tmp/cue_figures \
  --results-dir /tmp/cue_results
```

## Reproduce Model Scoring

The `thesis_framing_pipeline/` directory keeps the model-side pipeline used to
create the scored rows. It builds prompts, runs generation, judges stance, and
adds matched baseline effects.

Typical full run:

```bash
python thesis_framing_pipeline/00_validate_inputs.py
python thesis_framing_pipeline/01_build_prompts.py --mode pilot
python thesis_framing_pipeline/02_run_generation.py \
  --prompts thesis_framing_pipeline/results/prompts_pilot.csv \
  --out-jsonl thesis_framing_pipeline/results/generations_pilot.jsonl \
  --out-csv thesis_framing_pipeline/results/generations_pilot.csv \
  --device cuda:0 \
  --batch-size 8 \
  --max-new-tokens 1000
python thesis_framing_pipeline/03_run_stance_eval.py \
  --generations thesis_framing_pipeline/results/generations_pilot.csv \
  --out-jsonl thesis_framing_pipeline/results/evaluated_pilot.jsonl \
  --out-csv thesis_framing_pipeline/results/evaluated_pilot.csv \
  --device cuda:0 \
  --batch-size 16
python thesis_framing_pipeline/04_analyse_prelim.py \
  --evaluated thesis_framing_pipeline/results/evaluated_pilot.csv \
  --out-dir thesis_framing_pipeline/results/analysis_pilot
```

The local `data/processed/evaluated_with_effects.csv` is the scored output used
for the report figures. It is omitted from Git because it exceeds GitHub's
regular file-size limit. Regenerating it requires local access to the Qwen
models specified in `thesis_framing_pipeline/config.py`.

## Notes

- Survey weights use CES `commonweight`.
- CES state cue groups use CA/MA/NY as blue states, AL/OK/TX as red states, and
  GA/PA/WI as swing states, matching the prompt cues.
- The abortion item is ordinal and is mapped from CES codes `1..4` onto
  `[-1, +1]`; binary support/oppose items map support to `+1` and oppose to `-1`
  before applying each issue's liberal sign.
