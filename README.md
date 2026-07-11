# Identity-Cued Political Stance in Writing Assistance

A reproducible research artifact for an MSc thesis experiment: **when a writing-
assistance model is given a stored user-identity cue, does it change the political
stance it writes — and does that change track real-world opinion differences
between the corresponding groups?**

Model output and survey ground truth are put on one scale: a **liberal score** in
`{−1, 0, +1}` (−1 = writes the conservative side of the issue, +1 = the liberal
side, 0 = neutral). The same scale is reconstructed for the **CES 2025** survey so
the two are directly comparable.

Three research questions:

- **RQ1 — Effect.** Do identity cues shift the written stance, and by how much, by
  cue type (explicit party / explicit demographic label / implicit state / implicit
  name)?
- **RQ2 — Calibration.** Does the shift track the *real* CES subgroup−population
  difference (calibrated personalisation), under-shoot it (flattening), or overshoot
  it (stereotyping)?
- **RQ3 — Mechanism.** Behavioural + internal linear probes explaining *why* implicit
  (name) cues produce a near-null behavioural shift.

For the design, estimators, and findings in prose, see **`docs/`** (start with
`docs/methodology.md`, then `docs/robustness_checks.md`).

---

## Repository layout

```text
.
├── README.md                     # this file
├── docs/                         # methodology + findings + robustness (prose)
│   ├── methodology.md                # design, CES linkage, prompts, scorer, estimators
│   ├── robustness_checks.md          # the RQ2 robustness suite, written up
│   ├── analysis_decision_log.md      # pre-specified vs post-hoc decisions
│   ├── findings_cross_model.md       # cross-model results
│   └── probe_*.md                    # RQ3 probe methodology / findings / explainer
│
├── data/
│   ├── input/                    # committed experiment inputs (issues, templates, cues, names, states)
│   ├── reference/                # CES recoding metadata
│   └── processed/                # large built prompts + generations (gitignored)
│
├── pipeline/                     # MODEL SIDE — build → generate → score (numbered 00–10)
│   ├── 00_validate_inputs.py … 10_run_direct_probe.py
│   ├── <lib>.py                      # config, cues, prompting, sampling, stance, probe, io…
│   └── run_*.sh                      # drivers (local vLLM / HF, OpenAI batch)
│
├── stance_model/                 # DeBERTa-v3 stance cross-encoder: train / predict / metrics
│
├── analysis/                     # REPORT SIDE — numbered by analysis stage
│   ├── lib/                          # shared code (_common.py estimator, _regression.py helpers)
│   ├── 01_ground_truth/              # CES weighted subgroup estimates + descriptives
│   ├── 02_stance_scorer/             # classifier validation, LLM-judge agreement, score combine
│   ├── 03_cue_effects/               # cue-effect prep / non-neutral baseline
│   ├── 04_calibration/               # RQ2 calibration slope + stance-reduction sensitivity
│   ├── 05_robustness/                # the RQ2 robustness suite (+ run_robustness.sh)
│   ├── 06_probe/                     # RQ3 probe / PCT / legibility reports
│   └── plotting/                     # ALL figure generation (make_*.py); _legacy/ = superseded
│
├── results/                      # analysis outputs, CSVs (gitignored; local)
│   ├── full_3x/                      # DATA OF RECORD: 3-rep 2k-token DeBERTa scores + CES estimates
│   ├── robustness/                   # robustness-suite tables
│   ├── full/                         # earlier 1-rep run (raw generations + OpenAI arm + judge labels)
│   ├── probe_internal/  pct_*/  cue_probe*/
│   ├── stance_model_cv/              # cross-validated scorer predictions + metrics
│   └── _archive/                     # superseded outputs (gitignored, kept on disk)
│
└── figures/                      # rendered figures (gitignored; local)
    ├── full_3x/  full_bert/  robustness/  probe_thesis/
    └── _archive/                     # superseded figure iterations
```

## Data of record

The headline numbers come from **`results/full_3x/`**: a fresh **3-repeat,
2000-token** rerun of the three open-weight models, scored by the **DeBERTa-v3
cross-encoder** (`bert_liberal_score`). An earlier local LLM judge (Qwen) was a
placeholder and is retained only for validation; the OpenAI arm lives in
`results/full/`.

| Model | Access | Cue realizations |
|---|---|---|
| Llama-3.1-8B-Instruct | open weights (Brains GPU) | Arm A crossed + Arm B rotated |
| Gemma-3-12B-IT | open weights (Brains GPU) | ″ |
| Qwen3.6-27B | open weights (Brains GPU) | ″ |
| GPT-5.4-mini | OpenAI Batch API | ″ (confirmatory) |

**Design** (full detail in `docs/methodology.md`): 19 CES-linked IssueBench-style
issues; cue delivered as an inferred user memory in the system prompt (never in the
user turn); **Arm A** = fixed-condition cues (baseline + explicit party + explicit
demographic labels) crossed over 145 templates; **Arm B** = sampled-instance cues
(names, states) rotated over a genre-proportional 35-template subset.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # analysis deps
# CLMM robustness additionally needs R with the 'ordinal' package
```

CES respondent microdata (`CES25_Common.dta`) is **not committed**; scripts read it
from `../CES/CES25_Common.dta` (override the `CES_DTA` constant if elsewhere).

## Reproduce

**Model side (GPU, on Brains)** — build prompts, generate, score. See
`pipeline/README.md` for per-step commands and the `run_*.sh` drivers.

**Ground truth + calibration + robustness (local, from repo root):**

```bash
# 1. CES weighted subgroup estimates (the RQ2 x-axis)
python3 analysis/01_ground_truth/ces_estimates.py

# 2. stance-scorer validation (confusion matrix, per-class F1)
python3 analysis/02_stance_scorer/classifier_validation.py

# 3. the full RQ2 robustness suite (model-shift table → all checks → CLMM)
bash analysis/05_robustness/run_robustness.sh
```

`run_robustness.sh` builds `results/robustness/model_shift_table.csv` once (the
issue-clustered bootstrap backbone in `analysis/lib/_common.py`) and then runs every
check: Deming/free-intercept calibration slope, threshold + directional-only
sensitivity, TOST equivalence, composition/flattening, DiD variance propagation,
refusal Manski bounds, generation variance, instance breakdown, BH-FDR /
leave-one-issue-out / permutation, and the R CLMM. Outputs land in
`results/robustness/` and `figures/robustness/`.

## Key conventions

- Survey weights use CES `commonweight`; all CES means are survey-weighted.
- State cue groups: CA/MA/NY (blue), AL/OK/TX (red), GA/PA/WI (swing), matching the
  prompt cues.
- Binary items map support→`+1`, oppose→`−1`, then apply each issue's `liberal_sign`;
  the abortion item is ordinal, mapped from CES codes `1..4` onto `[−1,+1]`.
- Analysis scripts are run from the **repo root** (paths are root-relative). Scripts
  in numbered subdirs import shared code from `analysis/lib/` via a small path shim.
