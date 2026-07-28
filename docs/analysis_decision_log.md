# Analysis decision log

An MSc-scale honest substitute for preregistration: which analysis decisions were
fixed **by design, before seeing the DeBERTa-scored outcomes** (D), and which were
made **after** seeing results, as robustness responses to examiner-style critique
(R). Retroactive preregistration is impossible; this log is the disclosure.

## Fixed before seeing outcomes (by design)

- **(D) Unit and scale.** Per-response liberal score in {−1, 0, +1}; cue effect =
  cued mean − baseline mean; issue-averaged (equal weight per issue).
- **(D) Two-arm design.** Arm A fixed-condition cues crossed over 145 templates;
  Arm B sampled-instance cues (names/states) rotated over the genre-proportional
  ~35-template subset. Reps = 1 in the pilot, 3 in the full_3x rerun.
- **(D) CES linkage.** 19 main issues, survey-weighted (`commonweight`),
  subgroup − population shift as the ground-truth x-axis; subgroup definitions
  (`pid3`, `race`×`gender4`, state partisan class) fixed to match the cues.
- **(D) Stance classifier of record.** DeBERTa-v3 cross-encoder
  (`bert_liberal_score`); the Qwen judge is a placeholder retained only for
  validation.
- **(D) Calibration framing.** Model shift vs CES shift with the y = x line as
  perfect calibration; under-personalisation = slope < 1. The through-origin
  calibration slope was the pre-specified headline estimand.
- **(D) Bootstrap clustering.** Uncertainty from an issue-clustered resample.

## Decided after seeing results (robustness responses)

- **(R) Free-intercept fit** added alongside through-origin, to test the
  zero-real-difference → zero-model-shift assumption. (§ RQ2 regression)
- **(R) Deming / errors-in-variables slope** added because both axes are
  estimates; OLS attenuates. δ set from the ratio of mean error variances
  (CES issue/design SE vs model bootstrap SE).
- **(R) Per-model slopes + cue-clustered bootstrap** added because the 14×3
  points are not independent (each cue recurs once per model at identical x).
- **(R) TOST equivalence** on the name/state nulls, with the SESOI set to the
  real CES group difference (and half of it). Bound choice is the design's
  principled SESOI, fixed by the CES numbers, not tuned to the result.
- **(R) Composition / variance-collapse analysis** added to test the variance
  half of Wang et al.'s flattening claim (means alone cannot see it). Extremity
  gap and directional-share calibration were defined before computing them.
- **(R) DiD variance propagation** (add CES design variance to the model
  bootstrap variance).
- **(R) CLMM ordinal robustness** (cumulative link mixed model, crossed random
  effects) to defuse the ordinal-as-interval objection.
- **(R) Refusal Manski bounds, BH-FDR multiplicity, leave-one-issue-out,
  permutation test** — cheap insurance; none change any headline conclusion.

## Multiplicity posture (declared, not omitted)

The primary posture is **estimation, not testing**: effect sizes with CIs, with
interpretive claims made only about the large, replicated-across-models effects
(e.g. Republican ≈ −0.6) and the clean nulls (name cues). Marginal intervals
excluding zero are **not** interpreted as discoveries. As a companion for any
per-cue significance claim, the false discovery rate is controlled with
Benjamini–Hochberg across the 42-test cue×model family (`multiplicity_bh.csv`);
all naive-significant effects survive, because the findings are carried by large
effects and clean nulls rather than marginal ones.
