# RQ2 robustness checks

This document reports the robustness suite added to protect and sharpen the RQ2
calibration finding (does the model personalise toward *real* group differences?).
All checks run on the **data of record**: the 3-repeat, 2000-token rerun of the
three open-weight models (`results/full_3x/bert_eval_{llama,gemma,qwen}.csv`,
DeBERTa `bert_liberal_score` ∈ {−1, 0, +1}) against the survey-weighted CES 2025
ground truth. The per-(model, cue) model-shift estimator (issue-clustered
bootstrap, template-matched baseline for the rotated Arm-B cues) is defined once
in `analysis/lib/_common.py`; every check below reads from it.

Sections 1–6 follow the examiner memo's priority order: (1) regression form,
(2) equivalence, (3) composition/flattening, (4) DiD variance, (5) ordinal
robustness, (6) cheap insurance (refusals, multiplicity, LOO, permutation).
Sections 7–12 are the further checks: (7) threshold + directional-only stance
sensitivity, (8) generation variance, (9) instance-level breakdown, (10) classifier
validation, (11) genre heterogeneity, (12) CES coding + swing-state sensitivity. See
`docs/analysis_decision_log.md` for what was pre-specified vs added after results.

---

## 1. Calibration slope: through-origin vs free-intercept vs Deming

`analysis/04_calibration/rq2_regression.py` → `results/robustness/rq2_regression.csv`,
`figures/robustness/rq2_regression.png`.

The headline is the slope of model stance shift (y) on real CES group shift (x):
slope 1 = calibrated, 0 = ignores real differences, 0 < slope < 1 =
under-personalises.

| Scope | OLS through-origin | OLS free slope | Free intercept (p) | **Deming slope** (δ) |
|---|---|---|---|---|
| **Pooled** | 0.622 [0.18, 0.91] | 0.641 | **−0.035 (p=0.019)** | **0.751 [0.22, 1.00]** (δ=0.99) |
| Llama | 0.365 [0.13, 0.46] | 0.359 | +0.011 (p=0.39) | 0.375 [0.13, 0.47] |
| Gemma | 0.791 [0.24, 1.12] | 0.815 | −0.045 (p=0.076) | 0.902 [0.30, 1.14] |
| Qwen | 0.711 [0.14, 1.20] | 0.749 | −0.072 (p=0.021) | 0.877 [0.24, 1.19] |

Pooled CIs are a **cue-clustered bootstrap** (resample the 14 cues, each carrying
its 3 model points), which respects that the 14×3 points are not independent.

**Findings.**
- **Deming > OLS** (0.75 vs 0.62 pooled): correcting for x-measurement error
  de-attenuates the slope, as expected (δ ≈ 1, so the two axes carry comparable
  error and the correction is non-trivial). The under-personalisation conclusion
  **survives de-attenuation**: even the errors-in-variables slope is below 1, and
  its CI excludes 1 for the pooled and Llama fits.
- **Free intercept is small but significantly negative** pooled (−0.035, p=0.019;
  Qwen −0.072, p=0.021). A cue matching zero real difference produces a *slightly
  conservative* model shift — a common "any-memory" nudge, not toward liberal but
  toward the model's per-issue default. Worth one sentence in the write-up; it does
  not move the slope materially.
- **Model heterogeneity is large and interesting**: Llama's calibration slope
  (~0.37) is half Gemma's/Qwen's (~0.88–0.90). Llama under-personalises far more.
  Reporting per-model slopes is cleaner than pooling and surfaces this directly.

**Slope mediation — is it "just Republican"?** Leave-one-cue-out on the pooled fit:
dropping the Republican label alone takes Deming 0.75 → **0.41**; dropping both
party labels → **0.34**; among demographic/name/state cues *only*, the slope is
**~0.33**. The party labels supply almost all the x-axis leverage (Republican
x=−0.45, Democrat +0.40; every other cue lives within ±0.19). So calibration is
real for the strongest, most explicit signal (party) and close to flat for
demographic/implicit cues — under-personalisation is *worse*, not better, off the
party axis. Report per-family slopes, not only the pooled number.

## 2. TOST equivalence on the implicit-cue (name / state) nulls

`analysis/05_robustness/tost_names.py` → `results/robustness/tost_names.csv`.

A non-significant effect is not "no effect." TOST (Lakens 2017) against a
principled SESOI — the **real CES group difference** for the matching group (and
half of it) — upgrades the claim to *"we can reject effects as large as X."*
Two one-sided t-tests, df = 18 (issue clustering); equivalence = 90% CI inside
±bound.

- **Name cues: 9 / 12 (model × name) cells are statistically equivalent to zero
  within the full real CES difference.** The 3 exceptions are all **White-woman**
  names, whose *real* CES difference is itself ≈0.007 — there is essentially no
  real group gap to be equivalent to, and the effect is likewise ≈0.
- Within **half** the real difference, 11/21 implicit cells are still equivalent.
- **State cues genuinely move** (Gemma/Qwen red-state ≈ −0.07, tracking the real
  red-state shift of −0.072, not equivalent to zero). So the implicit-cue null is
  specifically about **names/demographics**, not implicit political signal — a
  sharper claim than "implicit cues do nothing."

## 3. Composition-level flattening (within-group variance collapse)

`analysis/05_robustness/composition_flattening.py` →
`results/robustness/composition_summary.csv`, `composition_per_issue.csv`,
`figures/robustness/composition_flattening.png`.

Wang et al.'s flattening includes **variance collapse** — treating a
heterogeneous group as a point — which a means-only analysis cannot detect. For
each cue group × issue we compare the model's **directional** (non-neutral)
liberal share to the real subgroup's liberal share (both among the opinionated,
which sidesteps the neutral-mapping problem). Collapse index = mean over issues of
(model extremity − CES extremity), extremity = |share − 0.5|.

- **44 / 45 cue×model cells show significant within-group variance collapse.** The
  model's directional responses sit at extremity ≈0.40 (nearly one-sided per
  issue) while the real groups sit at ≈0.14 (genuinely split). Mean collapse index
  +0.23.
- **The collapse is present at baseline** (no cue; index ≈0.245) and barely moves
  with the cue — i.e. it is the model's *default* per-issue one-sidedness, not
  something the identity cue induces. The means can look calibrated (slope 0.6)
  while the composition is collapsed regardless of cue.

This is the variance half of the flattening claim the study invokes, tested
directly, and among the first such audits to do so.

## 4. DiD variance propagation

`analysis/05_robustness/did_variance.py` → `results/robustness/did_variance_propagated.csv`,
`figures/robustness/did_calibration_propagated.png`.

DiDₖ subtracts a survey quantity from a model quantity; the published bootstrap
interval covered only the model term. We add the CES **design-based** sampling
variance (survey linearization, subgroup-vs-complement), independent of the model
term so the variances add.

- CES design SE is ≤0.009 for every subgroup — small relative to the effects, so
  calibration conclusions are unchanged.
- As anticipated, the **small Black subgroups** (n≈1000) widen the DiD interval by
  **+20–25%**; large subgroups by <7%. Better we widen them than an examiner does.

## 5. Ordinal robustness (CLMM)

`analysis/05_robustness/clmm_robustness.R` (ordinal::clmm) + `analysis/05_robustness/clmm_compare.py` →
`results/robustness/clmm_coefs.csv`, `clmm_vs_delta.csv`.

The mean liberal score treats conservative→neutral and neutral→liberal as equal
steps. As a scale-free check we fit a cumulative link mixed model,
`stance (conservative < neutral < liberal) ~ cue + (1|issue) + (1|template)`, per
model, and compare the ranking of the cue log-odds to the mean-difference Δₖ.

- Spearman ρ(CLMM log-odds, Δₖ): **Llama 0.94, Gemma 0.996, Qwen 0.996; pooled
  0.986** (all p < 1e-6). Pearson r ≥ 0.995 on every model.
- The cue coefficients rank-order the same as the Δₖ, so the RQ2 conclusions are
  **invariant to the ordinal-vs-interval scale assumption** (one appendix
  paragraph). (Lower sign-agreement on Llama, 71%, is only the near-zero name
  nulls flipping arbitrary sign on both scales — immaterial to the ranking.)

## 6. Cheap insurance

**Refusal Manski bounds** (`analysis/05_robustness/refusal_bounds.py`). Refusal rates are
0–5% and near-flat (Llama highest, ~4–5% on the name/state nulls; Gemma/Qwen ≈0).
Recomputing each Δₖ under worst-case refusal assignment (cued→+1/baseline→−1 and
vice-versa): **every cue effect with |Δ|>0.05 keeps its sign** (0/19 flip). Only
the near-zero nulls straddle zero, which threatens no positive claim. The name
null itself is robust: even reassigning all name-cue refusals to one extreme, the
bounds stay within ±0.05, still negligible vs the real 0.13–0.19 group gaps.
(Computed on the judge-labelled `full/` run, the only one retaining response text
locally; same {−1,0,+1} scale.)

**Multiplicity** (`analysis/05_robustness/rq2_extras.py`). Primary posture is estimation, not
testing (declared in the decision log). As a companion, BH-FDR across the 42
cue×model tests: 24/42 significant at naive p<0.05, and **all 24 survive FDR<5%** —
no marginal effects to lose, because the findings are carried by large effects and
clean nulls.

**Leave-one-issue-out.** Pooled slope is stable dropping any single issue:
OLS-origin ∈ [0.596, 0.657], Deming ∈ [0.714, 0.797]. No issue (abortion/climate)
drives it.

**Permutation test** (shuffle cue label within issue strata). 23/42 effects
significant at perm p<0.05 — a distribution-free companion agreeing with the
bootstrap.

## 7. Stance-reduction sensitivity: threshold + directional-only

`analysis/04_calibration/rq2_stance_sensitivity.py` →
`results/robustness/stance_sensitivity.csv`.

**Threshold sensitivity (B).** Recomputing every Δₖ and the slope under neutral
half-bands h ∈ {5, 10 (default), 15, 20}: **Republican is the strongest cue and the
name cues stay null in every band.** The slope magnitude drifts with the band
(OLS-origin 0.70 → 0.42 as the neutral zone widens) but stays < 1 throughout — the
qualitative conclusions are not a cutoff artifact.

**Directional-only reanalysis (C) — a headline reframe.** Dropping neutrals and
averaging over directional responses only (the forced-choice analogue of the CES
scale), the calibration slope jumps from 0.62 to **≈0.99** (Llama 0.54, Gemma 1.27,
Qwen 1.17). The mean-scale "under-personalisation" is therefore **substantially a
neutral-compression artifact**: when the model takes a side, its group-conditional
*direction* tracks real CES group differences almost 1:1 (and the strong models
overshoot the party cues). The model is not mis-directed — it hedges ("it depends")
far more than forced-choice survey respondents can. This coheres with the classifier
error structure (§10, support↔oppose confusion only 0.75%) and with the composition
finding (§3): directionally calibrated on the group mean, yet collapsed within-group
per issue.

## 8. Generation variance

`analysis/05_robustness/generation_variance.py` →
`results/robustness/generation_variance.csv`.

At temperature 0.7, are the effects sampling noise? Decomposing the per-response
liberal score with the (issue×template×cue) cell as the group: **74–81% of variance
is between-condition, ICC(1) 0.60–0.71.** Per-replicate cue effects (r01/r02/r03)
are stable — median rep-to-rep SD of Δₖ = 0.009; the large effects barely move
(Qwen Republican −0.645 / −0.642 / −0.657). The effects are systematic, not draws.

## 9. Implicit-cue instance-level breakdown

`analysis/05_robustness/instance_breakdown.py` →
`results/robustness/instance_effects.csv`, `figures/robustness/instance_breakdown.png`.

Defends the names-null against "unlucky name selection" (Tonneau et al.). Every
name/state behind the group estimate: name group means are all ≈0 (±0.02); the
per-name spread (~0.14 SD over ≈15 responses/name) matches pure sampling noise, so
**no individual name carries a real effect** — the null is about names as a class,
not a weak sample. States differ: red-state instances cluster tightly at a real
−0.06 to −0.08.

## 10. Stance-scorer validation

`analysis/02_stance_scorer/classifier_validation.py` →
`results/robustness/classifier_confusion.csv` (+ `results/stance_model_cv/cv_metrics.json`).

Everything rests on one scorer, so we document its error structure from the
cross-validated out-of-fold predictions (n=10,008 vs held-out human labels):
macro-F1 0.84, 3-class accuracy 0.89, per-class F1 = 0.93 (oppose) / 0.66 (neutral)
/ 0.94 (support). **Directional (support↔oppose) confusions are only 0.75%** — the
scorer almost never flips a side; its errors are concentrated at the neutral
boundary (neutral-band accuracy 0.67 vs 0.93 for clear sides). Its one weakness is
exactly the neutral axis that the directional-only analysis (§7) removes. (Human
codebook + inter-annotator κ and the LLM-judge triangulation remain as write-up
items; see `docs/stance_model_agreement.md`.)

## 11. Template-genre heterogeneity

`analysis/04_calibration/genre_heterogeneity.py` →
`results/robustness/genre_heterogeneity.csv`, `figures/robustness/genre_heterogeneity.png`.

Breaking the Arm-A cue effects out by the writing template's genre (pooled over
models): **every key cue keeps its sign across all genres** (Republican negative in
report through speech; Democrat/Black-woman positive throughout; White-man negative
throughout). The effect is not an artifact of one persuasive genre. Magnitude does
vary systematically — persuasive genres (speech, blog post, argument) amplify the
shift, informational ones (report, article) attenuate it — which is itself a useful
descriptive point for the discussion.

## 12. CES coding decisions + swing-state sensitivity

`analysis/01_ground_truth/ces_mapping_table.py` → `docs/ces_mapping.md`.

Documents the 19 per-issue coding decisions (CES variable, question item,
support/oppose codes, liberal-score mapping) and the state classification (2024
presidential margin). Reclassifying the borderline states (|margin| ≤ 3 pts) into
their lean moves the blue/red CES shifts by ≤0.008 and the residual swing class by
0.018; with model-side state effects near-null regardless, the calibration
conclusion does not depend on the contestable borderline assignments.

---

### Bottom line

The headline conclusions survive and sharpen. Under-personalisation on the mean
scale holds after de-attenuation (Deming 0.75 < 1) — **but the directional-only
reanalysis (§7) shows it is substantially a neutral-compression artifact**: when the
model takes a side, its group-conditional direction tracks reality almost 1:1
(slope ≈ 0.99). The calibration that does exist is carried by the explicit party
labels (slope ~0.33 without them). The name-cue null is now a *rejection of effects
as large as the real group difference* (TOST), robust to threshold, refusal, and
name selection; and the study documents **within-group variance collapse** the means
analysis could not see. Model heterogeneity (Llama ≈0.37 vs Gemma/Qwen ≈0.9) is real
and reported per-model. The correctness fixes (DiD variance, CLMM, refusal bounds,
FDR, LOO, generation variance, classifier validation) change no conclusion but
pre-empt the obvious objections.
