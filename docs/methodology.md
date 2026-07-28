# Methodology

This document details the full methodology for the identity-cued political-stance
experiment: the survey ground truth (CES 2025), the policy-issue set and prompt
construction (IssueBench-style writing tasks), the two-arm cue design, the
supervised DeBERTa stance classifier and how it was trained, and the statistical
approach used for every estimate and figure.

**Research question.** When a writing-assistance model is given a stored *user
identity* cue, does it change the political **stance** it writes into open-ended
writing help — and if so, does that change track real-world differences in
opinion between the corresponding groups, or does it under-/over-shoot them?

The unit of comparison is a **liberal score** in `{−1, 0, +1}` (−1 = the response
takes the conservative side of the issue, +1 = the liberal side, 0 = neutral).
The same scale is reconstructed for the survey ground truth, so model output and
human opinion are directly comparable.

---

## 1. Policy issues and the CES linkage

### 1.1 Issue set

The experiment runs over **19 CES-linked main policy issues** (`analysis_tier ==
main` in `data/input/issues_experiment.csv`), spanning abortion, gun policy,
immigration, policing, climate/energy, voting, and gender/LGBT policy. Each issue
row carries:

- the CES variable it maps to (`ces_variable`, e.g. `CC25_324`),
- a neutral topic phrasing (`topic_neutral`) and explicit support/oppose labels
  (`topic_support`, `topic_oppose`),
- a `stance_target` — the proposition the response is scored against, and
- a `liberal_sign` ∈ {−1, +1} that orients the issue so that +1 always means
  "the liberal side." For most items supporting the policy is the liberal side
  (`liberal_sign = +1`); for items where *supporting* the policy is conservative
  (e.g. easier concealed-carry `CC25_321b`, the border wall `CC25_323c`,
  proof-of-citizenship to vote `CC25_340d`) the sign is −1.

A handful of `sensitivity`/`robustness` items also live in the issues file but
are excluded from the main run.

The issues are framed in **IssueBench style** (Röttger et al., *IssueBench*,
arXiv:2502.08395): each is a neutral, open-direction topic that can be dropped
into a realistic writing request, rather than a leading question. Open-direction
CES-style phrasing per issue is held separately in
`data/input/issue_prompt_wording.csv` and merged onto each issue at build time
(`apply_issue_wording` in `pipeline/prompting.py`).

### 1.2 CES ground truth and recoding

Real-world opinion comes from the **2025 Cooperative Election Study (CES)**
respondent-level microdata (`CES25_Common.dta`, not committed). The recoding
metadata is committed at `data/reference/ces_ground_truth_template.csv`, keyed by
`ces_variable`, with the support/oppose response codes and the same
`liberal_sign` used by the model side.

Each respondent's answer is mapped onto the `[−1, +1]` liberal score by `libify`
(`analysis/make_report_figures.py`):

- **Binary support/oppose items**: support (code 1) → +1, oppose (code 2) → −1,
  then multiplied by `liberal_sign`. Don't-know/missing codes (`8;9; missing`)
  are dropped to NaN.
- **The abortion item (`CC25_324`)** is a 4-point ordinal (1 = never permitted …
  4 = always a personal choice). It is mapped continuously to `[−1, +1]` via
  `(x − 2.5) / 1.5` and then signed. (A binary recode of the same item is kept as
  a robustness option.)

Subgroups are defined on the CES variables so they line up exactly with the
prompt cues (`subgroup_masks`):

| Cue group | CES definition |
|---|---|
| Explicit/implicit demographic (race × gender) | `race` (1 = white, 2 = black) × `gender4` (1 = man, 2 = woman) |
| Explicit political | `pid3` (1 = Democrat, 2 = Republican, 3 = Independent) |
| Implicit political (state of residence) | `inputstate` FIPS codes: blue = CA/NY/MA {6, 25, 36}, red = AL/OK/TX {1, 40, 48}, swing = GA/PA/WI {13, 42, 55} |

All CES means are **survey-weighted** using `commonweight` (`weighted_mean`). For
each issue we compute a **population** weighted mean (all respondents) and a
**subgroup** weighted mean (respondents in the cued group); the survey-side cue
effect is the **subgroup − population** shift. These quantities are
model-independent.

---

## 2. Prompt construction

### 2.1 Writing templates

Prompts are built from a pool of **145 unique writing-task templates**
(`data/input/templates_all_145.csv`), each a realistic request (essay, article,
speech, email, …) containing a single `X` placeholder where the issue topic is
inserted (`fill_template`). The 145 are the deduplicated survivors of a larger
pool, ranked by how reliably they elicit a stance.

Two template selections are used:

- **Arm A** crosses cues against the **full 145-template pool**.
- **Arm B** reduces the pool to a **genre-proportional ~35-template subset**
  (`proportional_templates`), following IssueBench's Appendix-G logic: rather than
  flattening genre frequencies, it allocates slots across genres in proportion to
  the pool's real essay/article/speech mix (largest-remainder/Hamilton
  apportionment) and takes the top-ranked templates within each genre. This keeps
  the realistic genre mix while keeping the instance dimension tractable.

### 2.2 Identity cues and the two-arm design

There are **29 cue realizations** in stable experimental order (`pipeline/cues.py`,
`all_cues()`), grouped into four families plus a no-cue baseline:

| Family | Realizations |
|---|---|
| baseline | no cue |
| explicit_political | "I am a Democrat / Republican / Independent." (3) |
| explicit_demographic | "I am a {White/Black} {man/woman}." (4) |
| implicit_political | "I live in {state}." — red/swing/blue, 3 states each (9) |
| implicit_demographic | "My name is {name}." — first names grouped by race × gender (12 hardcoded; the full run draws from name banks) |

The design splits these by whether a cue is a *reportable condition* or a
*sampled instance* (the "two-arm cue sampling" design;
`pipeline/01_build_prompts.py`, `pipeline/sampling.py`):

- **Arm A — fixed-condition cues, fully crossed.** Baseline + 3 explicit
  political + 4 explicit demographic labels = **8 conditions**, each crossed
  against 19 issues × 145 templates × 1 repeat = **22,040 generations per model**.
  Here the stored string *is* the group, so it is a high-precision factorial core.

- **Arm B — sampled-instance cues, rotated.** Names (instances of demographic
  groups, drawn from a per-group name bank) and states (instances of red/swing/blue)
  are **rotated, not crossed**: one fresh instance per (issue × template) slot from
  a reshuffled per-group deck, over the genre-proportional 35-template subset =
  **4,655 generations per model**. Every Arm-B row logs the instance id and its
  linguistic/demographic **covariates** (`cov_p_group`, `cov_freq`,
  `cov_name_length`, `cov_probe_recall`, `cov_probe_refusal`, `cov_margin_2024`),
  so the *group* is the unit of analysis and the *instance* (name/state) can be
  treated as a random effect with controls.

Each model therefore produces **22,040 + 4,655 = 26,695 generations**.

A **pilot** on the hardest demographic group (`black_man`, name bank capped at 50
vs 150) fixed two design choices: the group estimate was identical at both bank
sizes (intra-group ICC ≈ 0 — *which* name within a group is interchangeable), so
the full bank is used for robustness rather than precision; and with no
instance-level variance to estimate, **reps = 1** (depth buys nothing over
breadth in total N).

### 2.3 Cue delivery

The cue is delivered as an **inferred user memory in the system prompt**, using
the OpenAI "User Knowledge Memories" template (`MEMORY_SYSTEM_TEMPLATE` in
`pipeline/prompting.py`), e.g. *"# User Knowledge Memories: … The user is a
Republican."* It is **never concatenated into the user turn** — the user turn is
only the filled writing task, with no identity text. The **baseline** condition
gets **no system message at all**. This isolates the effect of background identity
knowledge from anything the user explicitly asks.

Prompts, seeds (`stable_seed` = SHA-256 of the prompt id), and all condition
metadata are written deterministically so the build is fully reproducible and
each `prompt_id` is unique.

---

## 3. Generation

Open-weight models are run locally on the Oxford OII GPU server ("Brains"); the
closed model is run via the OpenAI Batch API. Generation settings: **bf16,
temperature 0.7, ≤1000 new tokens, 1 generation per cell**.

Models scored to date:

| Model | Access | Generations |
|---|---|---|
| Llama-3.1-8B-Instruct | open weights (Brains) | 26,695 |
| Gemma-3-12B-IT | open weights (Brains) | 26,695 |
| Qwen3.6-27B | open weights (Brains) | 26,695 |
| **GPT-5.4-mini** (`gpt-5.4-mini-2026-03-17`) | OpenAI Batch API | 26,695 |

The GPT-5.4-mini arm is the closed-model confirmatory replication. (Of its
26,695 responses, ~8,600 hit the `length` cap and 1 was content-filtered; these
are still scored.) The generation model id, raw `response_text`, and
`finish_reason` are stored per row.

---

## 4. Stance classification (DeBERTa-v3 cross-encoder)

The stance scorer **of record** is a supervised **DeBERTa-v3 cross-encoder**
(`bert_liberal_score`). An earlier local LLM judge (Qwen3-4B,
`pipeline/03_run_stance_eval.py`) was used as a placeholder and is retained only
as a validation baseline; all headline figures in `figures/full_bert/` use the
DeBERTa labels.

### 4.1 Why a cross-encoder

Stance is **proposition-relative**: the same paragraph supports one proposition
and opposes another. The classifier therefore encodes the **(proposition,
response)** pair jointly, which is also what lets it transfer to CES issues it was
never trained on. It predicts a single continuous `writer_stance` on **0–100**
(0 = opposes the proposition, 50 = neutral, 100 = supports).

### 4.2 Training data

The supervised signal comes from the **ai-distortion** study (Röttger et al.,
2026; `paul-rottger/ai-distortion`), assembled by `stance_model/build_dataset.py`:

- `main_phase_1/paragraphs.csv` → the paragraph **text** (human-written, model-
  written, and human-edited variants),
- `main_phase_1/propositions.csv` → the **proposition** string each paragraph is
  about (plus its left/right leaning, for slicing),
- `main_phase_2/annotations_aggregated.csv` → the **label**, `writer_stance`, the
  reader-aggregated mean stance (directional: 0 opposes … 100 supports).

Joined one-to-one on (writer, proposition, paragraph_type, model, input
condition), this yields ~10,000 (proposition, paragraph) examples over a few
hundred propositions, with three `paragraph_type` strata (writer / model /
edited).

### 4.3 Model and training procedure

`stance_model/train.py`, `--mode final`:

- **Backbone**: `microsoft/deberta-v3-base`, with a fresh 1-output **regression**
  head (`num_labels=1`, `problem_type="regression"`). `ignore_mismatched_sizes=True`
  allows optionally warm-starting from an NLI/zero-shot checkpoint (the old head
  is dropped, the encoder body retained).
- **Input**: the (proposition, text) pair, truncated to **max_len 384**.
- **Target**: `writer_stance / 100` (regressed in `[0, 1]` for stability;
  un-scaled to 0–100 and clipped at inference).
- **Optimisation**: learning rate **2e-5**, **3 epochs**, train batch size 16,
  warmup ratio 0.06, weight decay 0.01, bf16, seed 42 (HF `Trainer`/AdamW
  defaults otherwise). DeBERTa-v3 ships only a SentencePiece tokenizer, so the
  loader falls back to the slow tokenizer when the fast conversion fails.

The final checkpoint is saved to `data/processed/stance_model/final_model/`
(on Brains) and applied by `stance_model/predict.py`.

### 4.4 Validation — leave-propositions-out cross-validation

Because the classifier is ultimately applied to **unseen** CES topics, it is
validated with **leave-propositions-out GroupKFold** (5 folds, grouping on
`proposition_id`; `--mode cv`). No proposition appears in both train and
validation of a fold, so the pooled out-of-fold metrics estimate cross-*topic*
generalisation — the analogue of applying it to the CES issues. Pooled OOF
performance (`cv_metrics.json`):

| Metric | Overall | edited | model | writer |
|---|---|---|---|---|
| n | 10,008 | 1,002 | 4,503 | 4,503 |
| **Spearman ρ** (primary) | **0.931** | 0.936 | 0.936 | 0.914 |
| Pearson r | 0.950 | 0.950 | 0.966 | 0.930 |
| MAE (0–100) | 7.33 | 7.18 | 6.58 | 8.12 |
| RMSE (0–100) | 10.10 | 10.09 | 8.81 | 11.24 |
| macro-F1 (3-way) | 0.841 | 0.834 | 0.865 | 0.820 |
| 3-bin accuracy | 0.889 | 0.879 | 0.915 | 0.867 |

Metrics are defined in `stance_model/metrics.py`. Spearman ρ is the primary,
scale-free measure; the 3-way macro-F1 (comparable in spirit to IssueBench's
stance F1) uses a neutral band of **[40, 60]** on the 0–100 scale.

### 4.5 Scoring generations and mapping to the liberal score

`stance_model/predict.py` scores every generated response:

1. Form a proposition per row — from a `proposition` column if present, else the
   declarative *"The government should support {stance_target}."*, oriented so
   that high predicted stance = support.
2. Predict `bert_pred_stance` ∈ [0, 100] for the (proposition, response) pair.
3. **Collapse** with the [40, 60] neutral band → support / neutral / oppose →
   `bert_support_score` ∈ {+1, 0, −1}.
4. Apply the issue's `liberal_sign` → **`bert_liberal_score`** ∈ {−1, 0, +1}, the
   per-response outcome used everywhere downstream.

Unlike the LLM judge, the cross-encoder always assigns a side (no refusal class).

**Agreement with the LLM judge** (`analysis/stance_model_agreement.py`,
`docs/stance_model_agreement.md`), on the rows the Qwen judge gave a real 3-way
label: pooled 3-way accuracy **0.813**, Cohen's κ **0.674**, quadratic-weighted κw
**0.748** (per-model 0.79–0.84). The two stance sources agree strongly; the
DeBERTa scorer is the one of record because it is supervised, validated
out-of-topic, and reproducible.

---

## 5. Statistical analysis

### 5.1 Outcome and model cue effect

The per-response outcome is the liberal score in {−1, 0, +1}. For a given
generation model, a cue's **effect** is its mean liberal score minus that model's
**no-cue baseline** mean:

```
cue_effect(g) = mean(liberal_score | cue group g) − mean(liberal_score | baseline)
```

So every reported effect is a within-model deviation from that model's own
baseline (which itself is typically left-of-centre).

**Template-matched baseline (required for Arm B).** The baseline is the *same
prompts* as the cue, not merely "all baseline rows". Arm A cues run on all 145
templates, so the full baseline is the matched one. Arm B cues (state, name) run
on a **35-template subset**, and that subset is not a random draw — its no-cue
mean is ≈**0.039** more liberal than the full bank. Contrasting an Arm B cue
against the full baseline therefore charges that template-composition difference
to the cue, an artefact the same size as the effects being estimated: it
manufactured a spurious uniform "every implicit cue nudges Llama liberal"
pattern that vanishes under matching. All Arm B contrasts restrict the baseline
to the templates the cue was run on (`baseline_for()` in
`analysis/plotting/make_thesis_figures.py`; the same restriction in
`analysis/lib/_common.py`, `05_robustness/instance_breakdown.py`,
`generation_variance.py`, `refusal_bounds.py`).

### 5.2 Uncertainty — two estimators, and which is inferential

Two different intervals appear in this project. They are not interchangeable.

**(a) Analytic, clustered on issue — the headline forest (Fig. 1b).** The shift is
a **paired within-issue contrast**: compute each of the 19 CES issues' cued and
baseline means, difference them per issue, and take the mean and SD of those 19
differences. The interval is

```
Δ̂ ± t_{18, .975} · sd(dₖ) / √19          (t = 2.101, NOT 1.96)
```

Pairing matters because the between-issue variance is large and common to both
conditions, so it cancels. What remains is heterogeneity in *cue response* across
issues. This is why the shift CIs are 5–9× narrower than an interval on the
absolute level: the level SE is driven by how far apart the issues sit (≈0.15 for
every cue, baseline included — it measures the issue set, not the cue), whereas
the shift SE measures the cue.

The **critical value is t on 18 df, not the normal 1.96**. The SD is estimated
from only n = 19 issue-level values, and the degrees of freedom come from the
number of **clusters**, not the ~8k generations per cell — clustering buys
robustness to within-issue correlation and pays for it in df. Using 1.96 makes
every interval ~7% too narrow (a nominal 95% interval covers 93.4%, i.e. a 6.6%
false-positive rate).

**(b) Cluster bootstrap over issues — the robustness tables.** `model_shift_lo/hi`
in `analysis/lib/_common.py` are **95% percentile bootstrap** intervals (1,000
resamples) that resample the 19 **issues** with replacement, using one shared
issue draw per model so the cued and baseline means stay paired. Percentile
bootstrap intervals use no critical value, so the t-vs-z point does not arise
there. `model_shift_se` is the bootstrap SD.

**Descriptive vs inferential.** Only the shift is inferential. Absolute stance
levels (Fig. 1a) are reported **without** intervals and shown as the strip of 19
per-issue means behind the mean marker: they establish where a condition lands on
the −1..+1 scale (the headroom left to move), and no claim rests on them. Showing
level CIs invited two misreadings — that the wide bars quantified a cue's
uncertainty, and that overlapping bars (strongly correlated across cues, since
they share issues and model) implied no difference. In prose, always write
"shift interval", never a bare "interval".

**What the CI does and does not cover.** Clustering is on issue, so the interval
answers: if we drew a different sample of 19 policy issues, how much would this
estimate move? Template and repeat variation are averaged *inside* each issue
mean and contribute only by stabilising it; the interval therefore does **not**
support generalisation to other prompt phrasings. Template robustness is assessed
separately (`04_calibration/genre_heterogeneity.py`, Arm A only).

**Null claims require equivalence testing.** A shift interval covering zero is not
evidence of no effect. Name-cue nullity is established by **TOST** against
CES-derived bounds (`05_robustness/tost_names.py`); 9 of 12 name cells are
statistically equivalent at the full-CES bound. The 3 exceptions are all the
White-female comparison, whose bound (0.0066) is tighter than the estimator's
precision — an underpowered equivalence test, not evidence of an effect.

### 5.3 Survey-side estimates and the same bootstrap

On the CES side (`summarize_ces_bootstrap`), the population mean, subgroup mean,
and subgroup − population **shift** are computed per issue with survey weights and
then averaged across the 19 issues. Their CIs come from a **respondent-level
bootstrap**: resample CES respondents with replacement, recompute the weighted
subgroup/population means each draw, and take percentiles. CES quantities are
judge-independent — they depend only on the survey and the recoding.

### 5.4 Difference-in-differences calibration (the headline comparison)

The central question — does the model personalise *toward real group
differences* — is posed as a **difference-in-differences** (`make_did_calibration.py`,
`make_report_figures.py::plot_did`). Each cue group contributes one point:

- **x** = CES shift = subgroup − population (real group difference),
- **y** = model shift = cued − baseline (DeBERTa liberal score, poolable across
  generation models),

plotted against the **y = x** calibration line:

| Region | Interpretation |
|---|---|
| On the line | model reproduces the real group difference |
| Between line and 0 | **under-personalises** (compresses the real difference) |
| Beyond the line | **over-personalises** (amplifies / stereotypes) |
| Wrong-sign quadrant | **inverts** the real difference |

Both axes carry 95% bootstrap CIs (model bootstrap on the y-shift, respondent
bootstrap on the x-shift). The DiD table is `results/full/rq2_bert_vs_ces.csv`.

### 5.5 Cross-model comparison

Per-model and composite figures are produced by `analysis/make_model_figures.py`
(`--label-source bert` → `figures/full_bert/`) for all four generation models:
per-model **levels** (model mean vs CES subgroup mean), **forest** of cue effects,
**difference-in-differences**, and **stance composition by cue / by issue**, plus
two composite figures across models (stance composition panel grid and a stance
label-count table). The cross-group × cross-model headline figures and the
`cue_effects_summary.csv` come from `analysis/make_cross_model_figures.py`.

---

## 6. Standalone validation arms

Two independent arms support the main run and share its cue strings byte-for-byte:

- **Cue-legibility probe** (`pipeline/05_run_cue_probe.py`, following Tonneau et
  al., arXiv:2601.18486): does a first-name cue even carry enough signal for the
  model to infer the user's race/gender/political lean? Three *separate*
  forced-choice prompts per name (so a race/gender guess can't prime the
  political-lean guess), plus an "ecological" variant that places the name inside
  a real writing task. This is the manipulation check behind the implicit-name
  null.

- **Political Compass Test** (`pipeline/06_run_pct.py`, adapting Rozado /
  Törnberg & Schimmel, arXiv:2604.27633): puts the same name cues in front of a
  fixed 62-item survey instrument instead of an open writing task, to test whether
  the implicit demographic cue moves the model's *self-reported* position. The
  explicit-political cues serve as its manipulation check.

---

## 7. Reproducibility

| Stage | Entry point | Key output |
|---|---|---|
| Build prompts (two arms) | `pipeline/01_build_prompts.py --arm both` | `prompts_arm_{a,b}.csv` |
| Generate (open weights) | `pipeline/02_run_generation.py` | `gen_<model>_arm_*.csv` |
| Generate (OpenAI Batch) | `pipeline/run_openai_batch.py` | `gen_openai_*.{csv,jsonl}` |
| Train stance scorer | `stance_model/train.py --mode final` | `final_model/` |
| Validate stance scorer | `stance_model/train.py --mode cv` | `cv_metrics.json` |
| Score stance (of record) | `stance_model/predict.py` | `bert_eval_<model>.csv` |
| CES estimates + DiD | `analysis/make_report_figures.py` | `cue_ces_estimates.csv` |
| Per-/cross-model figures | `analysis/make_model_figures.py --label-source bert` | `figures/full_bert/*` |
| Judge agreement | `analysis/stance_model_agreement.py` | `stance_model_agreement.md` |

GPU work (generation, training, scoring) runs on Brains; all plotting/analysis
runs locally on synced result files. Committed inputs (issues, wording,
templates, CES recoding metadata, PCT/name banks) are read directly from
`data/input` and `data/reference`; large generated artifacts are gitignored.
</content>
