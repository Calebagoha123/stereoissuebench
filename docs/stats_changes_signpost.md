# Statistical changes: what changed, where it lands in the writing

Worklist companion to the 2026-07-30 statistics audit against the OII *Applied
Analytical Statistics* course (`paul-rottger/aas-2026-public`). Line numbers are `~/Desktop/SDS/thesis/{body,appendix}.tex` **at Overleaf commit `657027c`**;
the repo is an Overleaf mirror, so `git pull` and re-grep before trusting them.

Course grounding is given per item so the prose can echo the lectures' own wording
rather than paraphrase it. **No thesis prose has been written yet** — this file says
what to say and where, not the words.

---

## 0. What the course actually covers (so we cite it correctly)

| Week | Topic | Used for |
|---|---|---|
| 2 | i.i.d., LLN/CLT, SEs, analytical CIs, **bootstrap CIs** | every interval in the thesis |
| 3 | hypothesis testing, null distributions, p-values, tests for **means and proportions**, categorical association, **bootstrap testing** | cue effects, neutral-rate test |
| 4 | univariate OLS, one-sided tests, **Holm–Bonferroni** | calibration slope, multiplicity |
| 5 | multivariate OLS, statistical adjustment, conditional associations | cue × model interactions |
| 6 | logistic regression, GLMs, model fit | (was for refusal; now dropped) |
| 7 | potential outcomes, ATE, DAGs, backdoor criterion, **randomisation ⇒ exchangeability**; Poisson overdispersion, robust SEs | RQ1 causal claim, RQ3 post-treatment limitation |
| 8 | panel data, unit/time **fixed effects**, **clustered SEs**, random effects, serial correlation | the clustering rule, FE-vs-RE choice |

**Not in the course** (so each needs an explicit justification where used): equivalence
testing / TOST, errors-in-variables (Deming), ordinal / cumulative-link models,
Benjamini–Hochberg FDR, crossed random effects, two-way clustering.

---

## 1. Interval method: name cluster-*t* primary

**Status:** done. `analysis/05_robustness/ci_method_agreement.py` →
`results/robustness/ci_method_agreement.{csv,md}`.

**Finding:** the two methods agree on **70/70** cells. Point estimates differ by at most
0.0006. Cluster-*t* intervals are a median **1.10×** wider than the percentile bootstrap.
45 effects significant either way.

**Where it goes:** `body.tex:234` already describes both methods correctly — no
correction needed, just add that they agree on all 70 cells and that the cluster-*t* is
primary because it is the conservative one. Add the agreement line to the appendix
*Correctness checks* (`appendix.tex:~168`).

**Course wording to echo (W8, verbatim):** *"Violation of the independence assumption
leads to incorrect SEs and invalid inference. We fix this problem using clustered
standard errors that correct for within-unit correlation. Clustered SEs treat each unit
as an independent block of information: N clusters. Degrees of freedom are closer to N
than NT, which can create problems for small N."* That last clause is exactly why we use
$t_{18}$ and not 1.96 — say so; it is the strongest piece of course fidelity in the
thesis.

---

## 2. RQ1 is causally identified — currently under-claimed

**Status:** wording only, nothing to compute.

**Why:** the cue is *assigned*, crossed with issue and template. `body.tex:~139` already
says *"Holding the issue and writing template fixed, we vary only the user cue"* — that
is the identifying argument, it just isn't cashed out.

**Where it goes:** §3.1 (§Overview, around the "the treatment is a single cue" sentence)
and the opening of §4.2 (RQ1). State that $\widehat{\Delta}_k$ is an average treatment
effect, not an association.

**Course grounding (W7):** *"Randomisation guarantees exchangeability
$T \perp (Y^0, Y^1)$."* The DAG is `cue → stance` with issue and template blocked by
design. Say the backdoor criterion is satisfied trivially because there is no backdoor —
we control assignment.

---

## 3. Multiplicity: keep BH, but narrow the family and state the assumption

**Status:** decision made (BH stays); two textual additions needed.

**Numbers:** 70 tests. Naive $p<0.05$: **45**. BH FDR<5%: **44**. Holm–Bonferroni
FWER<5%: **29**.

> Correction to an earlier claim of mine: BH and Holm do **not** agree here. My "changes
> almost nothing" compared BH to *naive*. Holm would drop 15 effects, including
> GPT-5.6's Democrat effect (+0.068), Sonnet's Black-man label (+0.080), and every
> swing-state effect.

**Two things to add:**

1. **Why FDR, not FWER.** With 70 tests, family-wise control puts the per-test bar near
   $\alpha/70 \approx 0.0007$, which erases true effects of the size the race × gender
   claims rest on. The course itself frames the trade-off — Holm is introduced as
   *"uniformly more powerful → less increase in Type II error rate"* — so we are
   extending its own logic, not departing from it.
2. **Narrow the family to the 50 positive-claim cells** (party, race × gender, state).
   The 20 name cells argue *for* the null and are handled by TOST; including them
   inflates $m$ and spends power where a discovery is not wanted. Record this in the
   analysis-decision log so it reads as principle, not convenience.
3. **Acknowledge dependence.** BH controls FDR under independence or positive regression
   dependency. The 70 tests share issues and templates, so they are dependent —
   plausibly positively, which is the case BH tolerates. One sentence pre-empts the
   objection. (Benjamini–Yekutieli, the arbitrary-dependence version, would land near
   Holm.)

**Where it goes:** `appendix.tex:168` (*Multiplicity* in Correctness checks) for all
three; one clause in §4.2 where "significant" is first used.

---

## 4. Discretisation: justify on measurement, not on the slope it yields

**Status:** wording only. **Decision taken: CES-matching is the justification.**

**Why it matters:** the recorded reason for preferring `luna_liberal_disc` over the
continuous score was that the continuous version *did not reach a calibration slope of
1*. Choosing an outcome coding by the result it produces is a specification-search
problem and an examiner will press on it.

**What to say instead:** the discretised coding matches the **categorical form of the CES
response scale**, which is the comparison target — the survey records a direction, so the
model measure should too. Pre-specify it on that basis and report the continuous version
as a robustness check with whatever slope it gives.

**Where it goes:** §3.4 (Stance scoring, `body.tex:~188-200`) for the justification;
appendix *Stance-reduction sensitivity* (`appendix.tex:~106`) already carries the
threshold sweep — add the continuous-score slope there as a reported check.

---

## 5. Cue × issue calibration: fix the specification, keep the number

**Status:** done. `analysis/04_calibration/cue_issue_mixed.R` →
`results/robustness/cue_issue_mixed.csv`.

> Correction to an earlier claim of mine: I predicted the 0.15 conditional slope was a
> degenerate boundary fit. **It is not.** `isSingular()` is FALSE and the slope barely
> moves across specifications:

| Specification | $\beta$ | SE | 95% CI |
|---|--:|--:|---|
| current: `(1｜issue)+(1｜cue)+(1｜model)` | 0.146 | 0.042 | [0.064, 0.228] |
| model fixed: `(1｜issue)+(1｜cue)` | 0.146 | 0.042 | [0.064, 0.228] |
| **principled: model+cue fixed, `(1｜issue)`** | **0.123** | 0.043 | [0.040, 0.207] |
| all fixed, OLS (naive SE) | 0.119 | 0.043 | [0.036, 0.202] |

**Two changes to the appendix (`appendix.tex:145-155`):**

1. **A factual error to correct.** The text says *"with the issue component absorbing
   most variance and the cue component on the boundary."* That is **backwards**: cue
   carries the most variance (0.0275), issue almost none (0.0002), model 0.0017, and
   nothing is on the boundary.
2. **Re-specify and re-interpret.** Report the principled fit ($\beta = 0.123$) as
   primary: of the three grouping factors only **issue** is plausibly a sample of a
   population (19 policy items); cue (14 designed levels) and model (5 chosen systems)
   are fixed by design. Then explain *why* the slope is low rather than treating it as a
   worry: it is a **conditional, within-cue** slope, so cue-level calibration is carried
   by differences *between* cues, not by tracking the CES issue-by-issue within a cue.
   That is a substantive limitation of the calibration claim and worth stating plainly.

**Course grounding (W8):** random effects assume $\alpha_i \sim N(0, \sigma^2)$ — the
levels must be *draws*. Designed factor levels are not. This is the FE-vs-RE distinction
the lecture draws, applied to a crossed rather than a panel design.

---

## 6. Truncation: bound it on the corpus of record

**Status:** done. `analysis/05_robustness/finish_reason_flatness.py` →
`results/robustness/{finish_reason_flatness,truncation_bounds}.{csv,md}`.

**Why this exists:** `refusal_bounds.py` runs on `results/full/eval_*.csv` — the *earlier*
judge-scored run, three open-weight models — because refusal detection needs response
text and `full_3x` retains only scores. So the appendix's *"all run on the classifier of
record"* is inaccurate for that one check.

**What the new check establishes,** on the corpus of record, all five models, no text
needed:

- provider-side filtering is negligible (max 0.036% anywhere)
- truncation is the only real non-completion mechanism: Llama 0.05–0.30%, Gemma
  0.15–2.19%, **Qwen 2.75–5.41%**, frontier models 0%
- truncated responses score **more liberal** (+0.095 Llama, +0.228 Gemma, +0.082 Qwen),
  so truncation is *not* ignorable in principle
- but the bias it implies for any cue effect is at most **0.0027**, because the
  cue-to-baseline *difference* in truncation rate is small even where the rate is not
- **70/70** cue effects keep their sign under that bias (vs 56/70 under a worst-case
  $\pm 1$ Manski assignment, which is uninformative here because these responses were
  scored, not missing)

**Where it goes:** appendix *Correctness checks* (`appendix.tex:~168`), as a new sentence
in the *Refusals* item plus the truncation numbers. **Also scope the opening claim** at
`appendix.tex:~90` — say the refusal bounds run on the earlier corpus and why, rather
than implying the whole suite is on luna.

---

## 7. Predicted-vs-written: test against $\beta = 1$

**Status:** done. `analysis/06_probe/prediction_write_gap.py` →
`tables/predwrite.tex` (already `\input` at `body.tex:409`).

Replaces the earlier $|{\rm predicted}| - |{\rm written}|$ difference, which needed a
bias caveat. $\beta$ is signed, and $\beta = 1$ is the same reference RQ2's calibration
slope uses. Model now enters as a **fixed effect**, so $\beta$ is a within-model
transmission rate rather than a pooled slope over five non-exchangeable systems.

| Cue type | mean \|predicted\| | mean \|written\| | $\beta$ [95% CI] | BH $q$ ($\beta\neq1$) |
|---|--:|--:|--:|--:|
| Party label | 0.475 [0.441, 0.510] | 0.300 [0.250, 0.349] | 0.57 [0.48, 0.65] | <0.0002 |
| Race × gender | 0.292 [0.250, 0.334] | 0.097 [0.073, 0.121] | 0.21 [0.18, 0.24] | <0.0002 |
| State | 0.332 [0.304, 0.359] | 0.080 [0.063, 0.098] | 0.07 [0.05, 0.10] | <0.0002 |
| Name | 0.182 [0.153, 0.211] | 0.061 [0.051, 0.071] | 0.04 [−0.03, 0.10] | <0.0002 |

Contrasts: Party − State **0.50** [0.42, 0.57]; Race × gender − State **0.14**
[0.12, 0.17]; **State − Name 0.03 [−0.02, 0.09], p = 0.24 — not distinguishable.**

**Where it goes:** §4.4, the two paragraphs after Figure `fig:belief-stance`
(`body.tex:411-413`). The claims those paragraphs can carry:

- under-writing holds for **all four** cue types ($\beta$ CI excludes 1 everywhere)
- the gradient separates at the **top** (party and race × gender transmit more than
  state) but **not at the bottom** (state vs name is not separable on slope)
- what separates state from name is the **prediction** side: state draws label-sized
  predictions (0.332, above race × gender's 0.292) and name-sized writing. Names fail at
  *formation*; states fail at *transmission*.

---

## 8. Neutral rate: a live RQ3 claim does not survive

**Status:** done. `analysis/06_probe/neutral_rate_test.py` →
`results/probe_internal/neutral_rate_test.{csv,md}`.

Paired on issue (19 clusters, df = 18), same `[40, 60]` band applied to both sides so
the band cancels. The unpaired two-proportion $z$ is also reported, but the two sides are
not independent samples — the same cue × issue cells generate both — so the paired
interval is the one to quote.

| Model | written neutral | predicted neutral | difference [95% CI] |
|---|--:|--:|---|
| Llama-3.1-8B | 0.300 | 0.090 | **+0.210** [+0.134, +0.286] |
| Gemma-3-12B | 0.365 | 0.217 | **+0.149** [+0.069, +0.228] |
| Claude Sonnet 5 | 0.548 | 0.417 | **+0.131** [+0.054, +0.208] |
| Qwen-3.6-27B | 0.341 | 0.372 | −0.031 [−0.127, +0.064] — no difference |
| GPT-5.6 Terra | 0.250 | 0.640 | **−0.390** [−0.478, −0.303] — **reverses** |

**⚠ This contradicts the text.** `body.tex:407` currently asserts *"the predictions are
polarised while the writing hedges to neutral far more often."* That holds for three
models, fails for Qwen, and **reverses sharply for GPT-5.6**, whose *predictions* are far
more neutral than its writing (64% of its predictions land in the neutral band — it
hedges the profiling task, not the writing task).

**What to do:** scope the claim to the models where it holds, and separate two things the
sentence currently fuses — the **magnitude** shortfall ($\beta < 1$) is general; the
**neutral-composition** shift is not. A model can under-write in magnitude without
hedging its writing more than its prediction.

---

## 9. CLMM extended to five models

**Status:** running at time of writing (`clmm_robustness.R`, now all five models;
`clmm_input_{gpt56terra,sonnet5}.csv` generated). Was 42 of 70 cells (three open-weight
models only), which the appendix's *"fit per model"* implied was complete.

> Correction to an earlier claim of mine: the CLMM was **not** DeBERTa-stale. Verified by
> matching `clmm_input_llama.csv`'s baseline mean (0.566412) against `luna_liberal_disc`
> (0.566412) vs `bert_liberal_score` (0.380036). It has been on luna since 28 July.
> Two other things I flagged as missing were also already present: the CLMM already uses
> **crossed random intercepts** on issue and template, and the cue × issue disaggregation
> already uses **two-way cluster-robust SEs**. Nothing to add for either.

**Where it goes:** appendix *Ordinal robustness* (`appendix.tex:~135-143`) — update the
Spearman figures to cover five models, and drop or keep the per-model restriction
sentence depending on whether the frontier fits converge.

**Course grounding:** W5's own instruction — *"Model choice should follow from the
data-generating process"* — plus W6's GLM logic extended from two categories to three
ordered ones. The course stops at binary; ordinal is the same idea one step on, motivated
by a real violation (averaging $\{-1,0,+1\}$ assumes equal spacing between adjacent
categories).

---

## 10. RQ3 is descriptive for a *specific* reason — say which

**Status:** wording only.

The predicted opinion is a **post-treatment** variable: we assigned the cue and observed
two downstream outcomes. Correlating them identifies nothing about whether one causes the
other. That is W7's mediator/collider warning, and it is the precise reason RQ3 cannot
carry a mechanism claim — much better than the current vague hedging.

The same argument condemns the internal political-axis correlation more sharply than the
current text does (projection and written stance are both post-treatment).

**The fix, if you ever want RQ3 causal:** randomise the mediator — inject the prediction
into the prompt ("this user likely supports X") and see whether the writing moves. One
generation pass, all five models, no internals needed. Flagged, not proposed.

**Where it goes:** §4.4 opening (`body.tex:~375-390`, the "Both arms are exploratory"
paragraph) — replace the blanket hedge with the specific post-treatment argument, and
note which RQ3 claims *are* inferential (transfer above chance with a shuffled-label
control; the $\beta \neq 1$ tests) rather than giving all of them away.

---

## Files added / changed

| Path | What |
|---|---|
| `analysis/05_robustness/ci_method_agreement.py` | new — interval-method agreement (70/70) |
| `analysis/05_robustness/finish_reason_flatness.py` | new — non-completion + truncation bounds, all 5 models |
| `analysis/04_calibration/cue_issue_mixed.R` | new — FE-vs-RE specifications for the conditional slope |
| `analysis/06_probe/neutral_rate_test.py` | new — paired neutral-rate test |
| `analysis/06_probe/prediction_write_gap.py` | rewritten — $\beta$ vs 1, model fixed, BH |
| `analysis/05_robustness/clmm_robustness.R` | extended to 5 models |
| `analysis/05_robustness/run_robustness.sh` | wired in the new steps |
| `tables/predwrite.tex` (thesis repo) | regenerated |

## Pre-existing bug found while compiling (not caused by these changes)

`tab:calibration_slopes` is **referenced twice** — `appendix.tex:125` and
`appendix.tex:134`, both in the calibration-slope discussion — but
`tables/calibration_slopes.tex` is **never `\input`**. Both `\ref`s currently render as
`??` in the PDF. Verified present in the pre-merge Overleaf commit too, so it predates
this work.

Fix is one line (`\input{tables/calibration_slopes}` in the *Calibration slope: functional
form* subsection), but placement is a prose decision, so it is left for the writing pass.

Compile sanity: 58 pages before these changes, 60 after — the two extra pages are
`tab:predwrite` plus the two new §4.4 paragraphs. Nothing was lost in the Overleaf merge.

## Deliberately not done

- **Logistic regression for the direct-probe refusal.** Dropped: the refusal figure *and*
  prose are commented out in `body.tex:~430-458`, so there is no live claim to test.
  One loose end remains — `body.tex:453` still says the behavioural arm shows the model
  *"refusing, in the open, the inference the writing withholds,"* which rests on removed
  evidence.
- **Recomputing refusal Manski bounds on `full_3x`.** Impossible locally: response text
  is not retained for the three open-weight models. Would need a Brains pull — ask first,
  the corpus is large.
- **A rule-based refusal detector** for the two frontier models. Rejected: inventing a
  second, unvalidated refusal instrument to cover 2 of 5 models is worse than stating the
  gap.
