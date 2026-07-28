# Why implicit cues don't move stance — probe findings (3 open models)

A behavioral + internal probe study to explain the main-run result that **implicit
identity cues (first names, state of residence) barely shift the political stance
the model writes**, while explicit cues do. Design rationale and literature in the
chat thread; methods in [`methodology.md`](methodology.md). **Status: executed on
all three open-weight stance models — Llama-3.1-8B-Instruct, Gemma-3-12B-IT,
Qwen3.6-27B.** The pattern below replicates across all three. GPT-5.4-mini can do
the behavioral arm only (no activation access via API).

## Cross-model replication (behavioral = 4 models, internal = 3 open models)

| metric | Llama-3.1-8B | Gemma-3-12B | Qwen3.6-27B | GPT-5.4-mini |
|---|---|---|---|---|
| **A2** belief → stance correlation | 0.82 | 0.77 | 0.74 | 0.89 |
| **B3** internal political-axis → stance (mediation) | 0.82 | 0.74 | 0.92 | — (API) |
| **B1** label→name transfer (chance 0.25) | 0.69 | 0.49 | 0.63 | — (API) |
| **A3** perceived relevance of a *first name* (/100) | 3.9 | 0.9 | 2.3 | 6.9 |
| names: \|belief shift\| / \|stance shift\| | 0.11 / 0.02 | 0.18 / 0.02 | 0.10 / 0.01 | 0.06 / 0.01 |
| states: \|belief shift\| / \|stance shift\| | 0.35 / 0.04 | 0.37 / 0.04 | 0.42 / 0.04 | 0.15 / 0.03 |

Every model: writes toward its belief (r ≈ 0.74–0.89); the three open models encode
a name's demographic above chance (transfer 0.49–0.69) and show the **states =
strong-belief-but-no-use** dissociation; all four rate a first name as near-useless
(≤ 7/100). GPT (API, behavioral only) matches the behavioral pattern. Numbers in the
sections below are Llama (the worked example); the others match.

Scripts: `pipeline/07_run_belief_probe.py` (A2/A3), `pipeline/08_extract_activations.py`
(B), `analysis/train_identity_probe.py` (B1/B2/B3), `analysis/make_probe_figures.py`.

## The explanatory ladder

The null decomposes into a causal chain; each probe targets one rung:

1. **Legibility / internal encoding** — does the model read the attribute from the cue?
2. **Belief** — does it form an opinion-prior about the user?
3. **Relevance** — does it treat the attribute as diagnostic?
4. **Use** — does it write to that belief?

## Headline result

**The "no adjustment" null is mostly a *use* gap, not an *inference* gap.** Llama
internally encodes who the user is from implicit cues, and *when it forms a belief
it acts on it* — but it forms weak beliefs from names and declines to act on the
(strong) beliefs it forms from states.

| | names (implicit demo.) | states (implicit pol.) | explicit labels |
|---|---|---|---|
| attribute decodable internally (B1) | **yes** (transfer 0.69, chance 0.25) | yes (1.00) | yes (1.00) |
| forms belief about user (A2) | weak (0.11) | **strong** (red −0.54) | moderate (0.29) |
| rates attribute diagnostic (A3) | **no** (name 3.9/100) | partly (state 39) | race/gender 24 |
| shifts written stance | ~0 (0.02) | ~0 (0.04) | small (0.07) |
| **failure mode** | low relevance → low belief → low use | **belief without use** | (acts, weakly) |

## A2 — the model acts on its beliefs, but under-acts (behavioral)

Conditioned on each cue, we asked the model to predict the user's probability of
supporting each of the 19 CES policies (0–100), centred/signed onto the liberal
axis. Across the 14 cue groups, the model's **belief shift predicts its written
stance shift at r = 0.82** (`figures/probe/llama_belief_vs_stance.png`) — so the
machinery is "form a belief about the user, then write toward it."

But it **systematically under-acts**: it believes a Republican-cued user opposes
the liberal side by −0.80 yet shifts its own writing only −0.22; every point sits
well inside the y=x line. This is the compression/under-personalisation the DiD
calibration also shows, now traced to the belief→output step.

The two implicit families fail **differently**:
- **Names**: weak belief itself (mean |belief shift| 0.11 vs 0.29 for explicit
  labels) → weak stance (0.02). Inference-/relevance-limited.
- **States**: *strong* belief (blue +0.46, red −0.54) but near-zero stance shift
  (0.04). A clean **belief-without-use** dissociation.

## A3 — the model rates a first name as near-useless (behavioral)

Self-rated usefulness for predicting opinion, mean over 19 issues
(`figures/probe/llama_relevance.png`):

```
political party 77  ≫  U.S. state 39  >  gender 25 ≈ race 24  ≫  first name 4
```

The model rates a **first name 3.9/100** — essentially uninformative — even though
it rates race/gender ~24 and a name *carries* race/gender. That meta-level gap
(name ≪ the attributes it encodes) mirrors the weak name adaptation exactly, and
party's dominance matches political cues driving the largest stance shifts.

## B1 — a name's demographic *is* encoded, like an explicit label (internal)

We extracted residual-stream activations (33 layers, last prompt token) over
cue × issue × template and trained linear probes (StandardScaler → PCA-256 → L2
logistic regression), GroupKFold over issues, shuffled-label controls.

The decisive test is **cross-cue transfer**: train a race × gender probe on
**explicit-label** rows, test it on **name** rows
(`figures/probe/llama_transfer.png`). It reaches **balanced accuracy 0.69**
(chance 0.25) — the explicit "Black woman" direction substantially fires for "My
name is Aaliyah". So **the model internally represents the name's demographic,
sharing a direction with explicit labels**, even though it barely acts on it and
calls names uninformative. Within-subset decodability hits 1.0 for every family
(state, party, explicit demographic, name) with high selectivity over the
shuffled controls — but for names within-subset reuses only 3 names/group, so the
*transfer* number is the honest "reads the name as a group" estimate.

→ For names the break is **after** attribute encoding: name → demographic is
encoded; demographic → political-opinion → output is where it attenuates.

## B2/B3 — internal political position mediates the written stance (internal)

We built a liberal↔conservative axis as the Democrat−Republican mean-activation
difference per layer and projected every group onto it. Across cue groups, the
**internal political-axis shift predicts the written stance shift at r = 0.82**
(`figures/probe/llama_mediation.png`): groups that move along the Dem−Rep
direction internally are the ones whose output moves. Explicit political cues move
furthest along the axis; implicit cues move little — the same gradient as the
output. So the political axis is the bottleneck implicit cues fail to reach in
force, despite their demographic content being encoded.

## Takeaway for the narrative

Both of the "either/or" outcomes are true, in different places:
- **Models *do* read political/demographic identity from implicit cues** — names
  are linearly decodable (transfer 0.69), states perfectly (1.0), and beliefs are
  non-zero. It is **not** primarily an inference failure.
- **Yet they largely don't adapt output**, because (i) for **names** the inferred
  identity is weakly believed and explicitly deemed non-diagnostic (relevance 4),
  and (ii) for **states** a strong belief simply isn't propagated to the written
  stance. The mediator is the internal political axis (r = 0.82 both for belief→
  stance and projection→stance), which implicit cues reach only weakly.

## Direct refusal probe — refusal is selective by *sensitivity*, not cue clarity

A separate neutral-framed probe (`pipeline/10_run_direct_probe.py`,
`direct_probe.py`) asks plainly — "a user has the first name X, what is their most
likely gender / race / political leaning?" — with **no** anti-refusal scaffolding,
so natural refusal can surface. Commit vs refuse rate (3 open models):

| attribute | commit | refuse |
|---|---|---|
| gender (Bob→male, Mary→female) | **1.00** | 0.00 |
| race | 0.43–0.63 | 0.35–0.48 |
| **political leaning** | 0.17–0.30 | **0.70–0.83** |

Gender is answered every time (~91–100% correct); refusal rises with the
attribute's social sensitivity, peaking on politics. So models **don't** refuse
because a cue is unclear — they refuse the *sensitive inference*. This is the
behavioral counterpart of the use-gap: the model declines, when asked outright, to
map a name onto a political opinion — the very step that's missing in its writing.

## Causal test (steering) — inconclusive on first pass

We added the Dem−Rep direction back into the residual stream at mid layers
(`pipeline/09_steer_generate.py`) and re-scored. The text stayed coherent but
written stance did **not** move monotonically with the steering coefficient
(corr ≈ −0.04), with a hint of an effect on stance-eliciting templates only. This
does not refute the axis being causal (the predictive mediation r=0.74–0.92
stands); it means simple mean-difference steering here is under-powered/noisy.
A clean test needs stance-eliciting templates only, more issues/templates for
power, a single-layer magnitude sweep, and possibly a topic-general
liberal–conservative direction rather than the cue-token mean difference.

## Next

- ✅ Gemma-3-12B + Qwen3.6-27B done — pattern replicates (table above). Note
  Gemma-3 residual-stream outliers overflow fp16, so activations are stored in
  float32 (`pipeline/08_extract_activations.py`).
- GPT-5.4-mini: A2/A3 via the OpenAI batch path (behavioral only; no internals).
- Optional causal test: steer along the B2 political axis at generation time
  (à la Neplenbroek) to confirm the axis is causal, not just predictive.

## Artifacts

- Behavioral A2/A3: `results/full/belief_probe_{llama,gemma,qwen,gpt}.csv`
- Direct refusal probe: `results/full/direct_probe_{llama,gemma,qwen}.csv`
- Causal steering: `results/probe_internal/steer_llama_scored.csv` (+ `data/processed/probe_activations/llama_steer_dirs.npz`)
- Internal: `results/probe_internal/{tag}_{decodability_by_layer,cross_cue_transfer,political_projection,mediation,summary}.*`
- Figures: `figures/probe/{tag}_{relevance,belief_vs_stance[,transfer,mediation]}.png`
- Activations (gitignored): `data/processed/probe_activations/{tag}_acts.npz` (+ on Brains `/data/kell8360/probe_activations`)
