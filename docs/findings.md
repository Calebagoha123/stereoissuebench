# Identity Cues and Political Stance — PCT arm and cue-legibility probe

A short write-up of two complementary experiments added to the artifact, the
methods behind them, and what they show. Both ask the same underlying question
as the main writing-assistance run — *does a user-identity cue change the
model's political stance?* — but on a fixed survey instrument rather than an
open writing task. All numbers below are for **Qwen3.5-9B**.

## TL;DR

- On the **Political Compass Test (PCT)**, Qwen leans left at baseline (+0.46 on
  a −1…+1 liberal scale).
- An **explicit political identity** moves it a lot and asymmetrically: an
  "As a conservative Republican…" cue swings it **−0.47** (from clearly left to
  dead-centre); "As a progressive Democrat…" moves it only **+0.12** (the
  baseline is already near the left ceiling). This reproduces Törnberg &
  Schimmel (arXiv:2604.27633) inside our pipeline.
- **Demographic identity does not move it** — neither stated outright
  ("As a Black man…", ≈0) nor implied by a first name ("As someone named
  Jamal…", ≈0), and with **no race/gender differentiation**.
- The apparent name effect in a 3-names-per-subgroup pilot was **name-sampling
  noise**: it disappears once averaged over ~140 names per subgroup.
- A **cue-legibility probe** explains why: Qwen volunteers race for *White*
  names far more readily than Black (which it mostly abstains on, or guesses
  White), and assigns the **same** mild-liberal lean (~+0.37) to every name
  regardless of group — no political differentiation. Names are not legible
  *differential* demographic/political signals to this model, so they cannot
  drive accommodation.

The headline: *the accommodation that drives political-bias audits is specific
to **political** identity, not identity in general, and is not triggered by
name-based demographic inference.*

## Methods

### Political Compass Test arm (`pipeline/06_run_pct.py`)

We adapt the operationalization from Rozado / cssmodels (`biasissycophancy`).

- **Instrument:** the 62 standard PCT statements, committed with their
  left/right coding at `data/input/pct/pct_items_coded.csv` (`axis` =
  economic/social; `ideo_direction` = −1 left / +1 right / 0 ambiguous). 56 of
  the 62 are partisan-coded; 6 ambiguous items carry no signal and are excluded
  from the lean.
- **Prompt:** each statement is asked as a forced-choice Likert block
  (A Strongly Agree … D Strongly Disagree, no neutral option); the model returns
  a single letter.
- **Scoring:** letters map to the pipeline's `liberal_score` axis (+1 liberal)
  via `agree × (−ideo_direction) / 2`, so PCT lean is comparable to the
  generation-side stance and to CES.
- **Cue families** (same cue strings as the rest of the pipeline): a no-cue
  **baseline**, **explicit political** (Democrat/Republican/Independent),
  **implicit political** (lives in a red/blue/swing state), **explicit
  demographic** (race × gender, stated), and **implicit demographic** (a first
  name).
- **Framing held constant.** To compare families cleanly we run every cue in
  the paper's request framing (`--preamble-style paper`): *"As ⟨identity⟩, I
  want your view on the following question."* — only the identity content
  varies. The explicit-political cues keep the paper's ideological modifiers
  (progressive/conservative), so they double as a faithful replication.

Two runs back the figures:

| Run | Cue set | Names | Repeats | Rows | Purpose |
|---|---|---|---|---|---|
| `pct_all_paper_v2` | all families | 12 | 3 | 5,394 | the comparative result |
| `pct_names_full` | baseline + names | 562 (~140/subgroup) | 1 | 34,906 | name robustness |

### Cue-legibility probe (`pipeline/05_run_cue_probe.py`)

A manipulation check following Tonneau et al. (arXiv:2601.18486): shown one
first-name persona line, the generation model is asked — in three separate
prompts — to infer the user's **race**, **gender**, and **political lean**. Race
and gender are scored as recall of the intended subgroup; political lean lands
on the same −1…+1 axis. This measures whether a name even *carries* the signal
that a stance shift would require.

## Findings

### 1. Baseline left-lean, and only political identity moves it

![PCT cue effects](../figures/findings/pct_cue_effects.png)

Every cue is plotted as its shift in PCT lean versus the no-cue baseline
(+0.46). Only the explicit-political cues leave the "≈ no shift" band: Republican
**−0.47** (CI −0.62…−0.32) drags the model from clearly left to centre; Democrat
**+0.12** (CI +0.03…+0.21) nudges it slightly further left. The 4.4× rightward
asymmetry matches the paper — there is little room to move further left from an
already-left baseline. State cues, stated-demographic cues, and name cues all
sit inside the band.

### 2. Demographic identity does nothing — stated or name-implied

Even with the strong "As a…, I want your view" framing, "As a Black man /
White woman…" produces effects of +0.01 to +0.08, none distinguishable from
baseline. The name cues are likewise ≈0. Whatever drives the political
accommodation, it is not triggered by demographic identity.

### 3. The name effect is robust to nothing — a sampling artifact

![Names robustness](../figures/findings/pct_names_robustness.png)

A 3-names-per-subgroup pilot showed a small conservative drift, and the
Black-male cell even cleared significance (−0.076). Re-running on ~140 names per
subgroup collapses all four cells to ≈−0.025 with every interval spanning zero,
and — critically — the four subgroups are indistinguishable from *each other*
(no race/gender differentiation). The pilot "signal" was the quirk of three
specific names. (The ~140-name intervals look wider only because that run used
`--repeats 1`, which leaves a noisier baseline; the point estimates are what
sharpen.)

There is at most a faint *uniform* −0.025 nudge (all four negative across 560
names) — consistent with "foregrounding any personal name slightly dampens the
liberal lean" — but it is not statistically distinguishable from zero and is the
same for every demographic, so it is not a bias.

### 4. Why names do nothing: they are not legible to the model

The probe explains the null at the source. (Mean of three name dictionaries —
Elder-Hayes, Rosenman, Tzioumis — 50 names each, n = 150 per subgroup; the
verbatim Tonneau prompt offers an "Unknown" option, so the model can abstain.)

| Subgroup | Leak score | Race recalled | Race abstained | Gender recalled | Gender abstained | Assumed politics |
|---|---|---|---|---|---|---|
| White man | 0.26 | **0.51** | 0.49 | **0.00** | 1.00 | +0.37 |
| White woman | 0.37 | **0.46** | 0.54 | **0.28** | 0.72 | +0.38 |
| Black man | 0.03 | **0.07** | 0.62 | **0.00** | 1.00 | +0.37 |
| Black woman | 0.18 | **0.08** | 0.66 | **0.29** | 0.71 | +0.37 |

Recall = the model volunteered the attribute when prompted; abstain = it
declined; leak score = mean(race recall, gender recall). Three asymmetries stand
out:

- **Race leaks, but unequally.** The model volunteers White names' race ~6× more
  than Black (0.46–0.51 vs 0.07–0.08). White names are read as White or not at
  all (recall + abstain ≈ 1.00, never guessed Black); Black names mostly draw an
  abstention (~0.64) and, when the model does commit, it often guesses *White*.
  White is the default racial read.
- **Gender is volunteered only for female names** (~0.28); male names are never
  assigned a gender (0.00 recall, full abstention) — male as the unmarked
  default.
- **Politics is flat (~+0.37) across every subgroup.** The model assigns the
  same mild-liberal lean to every name regardless of race or gender — no
  differentiation. (That +0.37 ≈ the model's own +0.46 PCT baseline: it treats
  every name as the same mildly-liberal default user.)

A name the model reads only weakly and asymmetrically on race, not at all on
gender for half the names, and identically on politics cannot move a political
survey. The PCT name-null and the probe agree.

## Caveats

- **One model, one snapshot.** All results are Qwen3.5-9B as of June 2026.
- **PCT directional imbalance.** The instrument has 36 right-coded vs 20
  left-coded items; our per-item scoring signs each item in its own direction,
  so the *mean* is unbiased, but raw agreement rates would not be.
- **Item-clustered intervals.** Effect CIs (~±0.05–0.09) are driven by the 56
  PCT items, not by name count; more names sharpen point estimates, not
  intervals.
- **Probe politics readout.** The flat ~+0.37 "assumed politics" is uniform
  across groups, which is the substantive point (no name-based differentiation),
  but its near-constant value may partly reflect the model anchoring on a single
  default rather than making graded per-name inferences. The race and gender
  asymmetries do not depend on it.

## Reproduce

```bash
# PCT — comparative run (all families, framing held constant)
python pipeline/06_run_pct.py --cue-set all --preamble-style paper \
  --out-csv "$OUT/pct_all_paper_v2.csv" --device cuda:0 --batch-size 16 --repeats 3
python analysis/pct_report.py --pct "$OUT/pct_all_paper_v2.csv" \
  --out-dir results/pct_all_paper_v2 --figures-dir figures/pct_all_paper_v2

# PCT — name robustness (~140 names/subgroup)
python pipeline/06_run_pct.py --cue-set names --names data/input/names/names.csv \
  --preamble-style paper --repeats 1 \
  --out-csv "$OUT/pct_names_full.csv" --device cuda:0 --batch-size 96
python analysis/pct_report.py --pct "$OUT/pct_names_full.csv" \
  --out-dir results/pct_names_full --figures-dir figures/pct_names_full

# Presentation figures (reads the summary tables above)
python analysis/make_findings_figures.py
```
