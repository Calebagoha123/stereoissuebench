## Non-completion on the corpus of record (`full_3x`, luna scoring)

Rates of provider-side filtering and token-cap truncation, per model, over the 15 conditions (14 cues + baseline). This checks whether *visible* non-completion is flat across cues; it cannot see a soft refusal that terminates normally, which is why the text-based Manski bounds (`refusal_bounds.py`, earlier corpus, three open-weight models) remain the binding refusal evidence.

| Model | conditions | filtered: max | truncated: min-max | truncated: spread |
|---|--:|--:|--:|--:|
| Llama-3.1-8B | 15 | 0.000% | 0.05%-0.44% | 0.39 pp |
| Gemma-3-12B | 15 | 0.000% | 0.15%-2.19% | 2.04 pp |
| Qwen-3.6-27B | 15 | 0.000% | 2.75%-5.41% | 2.67 pp |
| GPT-5.6 Terra | 15 | 0.036% | 0.00%-0.00% | 0.00 pp |
| Claude Sonnet 5 | 15 | 0.000% | 0.00%-0.00% | 0.00 pp |

- highest filtering rate anywhere: **0.0363%** (5 of 75 conditions show any at all)
- highest truncation rate anywhere: **5.41%** (Qwen-3.6-27B, implicit_political/blue_state)
- largest within-model spread in truncation across conditions: **2.67 pp**

Provider-side filtering is essentially absent. Truncation is the only non-completion mechanism with a non-trivial rate, so it is bounded below rather than assumed away.

### Worst-case bounds under adversarial truncation

Every truncated response is treated as unobserved and assigned the most adversarial value on the $\{-1,0,+1\}$ scale (cued $\to +1$ / baseline $\to -1$ for the upper bound, and the reverse for the lower). This is the same Manski logic `refusal_bounds.py` applies to refusals, but it runs on the corpus of record for **all five models**, since `finish_reason` needs no response text.

| Model | cues | sign robust | widest bound | not sign-robust |
|---|--:|--:|--:|---|
| Llama-3.1-8B | 14 | 12/14 | 0.0171 | imp/white_man (-0.007), imp/swing_state (+0.001) |
| Gemma-3-12B | 14 | 12/14 | 0.0854 | exp/white_woman (-0.038), imp/blue_state (-0.012) |
| Qwen-3.6-27B | 14 | 4/14 | 0.2246 | exp/black_man (+0.035), exp/white_woman (-0.032), exp/independent (-0.086), imp/black_man (-0.027), imp/black_woman (-0.023), imp/white_man (-0.032), imp/white_woman (-0.011), imp/blue_state (-0.036), imp/red_state (-0.078), imp/swing_state (-0.047) |
| GPT-5.6 Terra | 14 | 14/14 | 0.0000 | -- |
| Claude Sonnet 5 | 14 | 14/14 | 0.0000 | -- |

- across all 70 cue effects, **56** keep their sign under worst-case truncation
- restricting to the 35 effects with $|\Delta| > 0.05$ (the ones carrying claims): **33/35** are sign-robust
- the widest bound anywhere is **0.2246** on the $-1$ to $+1$ scale

### Why the worst case is the wrong bound here

Truncated responses were still *scored* -- they are not missing data. So the worst case, which assigns every truncated response $\pm 1$, answers a question the design does not pose. What the data supports is a point estimate of the contamination: how much truncation shifts a score, times how much more often it happens under the cue than under the baseline.

| Model | trunc rate | score shift when truncated | max implied bias | sign robust |
|---|--:|--:|--:|--:|
| Llama-3.1-8B | 0.05-0.30% | +0.095 | 0.0005 | 14/14 |
| Gemma-3-12B | 0.15-2.19% | +0.228 | 0.0027 | 14/14 |
| Qwen-3.6-27B | 2.75-5.41% | +0.082 | 0.0017 | 14/14 |
| GPT-5.6 Terra | 0.00-0.00% | +0.000 | 0.0000 | 14/14 |
| Claude Sonnet 5 | 0.00-0.00% | +0.000 | 0.0000 | 14/14 |

- truncated responses score more liberal than completed ones in every model that truncates at all (shift $+0.000$ to $+0.228$), so truncation is **not** ignorable in principle
- but the bias it implies for any cue effect is at most **0.0027**, because the cue-to-baseline difference in truncation rate is small even where the rate itself is not
- under that bias, **70/70** cue effects keep their sign, against 56/70 under the worst case

So the worst-case failures are an artefact of the $\pm 1$ assignment, not evidence of a truncation problem: the largest bias the observed data implies (0.0027) is an order of magnitude below the smallest cue effect the thesis interprets. Report the plausible-case column as the finding and the worst case as the (uninformative) extreme.
