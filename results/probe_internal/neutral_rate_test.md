## Neutral rate: written stance vs elicited prediction (band [40, 60] on both sides)

Paired on issue (19 clusters, df = 18); cued conditions only. A positive difference means the writing is neutral more often than the prediction is.

| Model | written neutral | predicted neutral | difference [95% CI] | t | p (paired) | p (unpaired z) |
|---|--:|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 0.300 | 0.090 | +0.210 [+0.134, +0.286] | 5.8 | 1.71e-05 | 2.26e-120 |
| Gemma-3-12B | 0.365 | 0.217 | +0.149 [+0.069, +0.228] | 3.9 | 9.99e-04 | 2.03e-55 |
| Qwen-3.6-27B | 0.341 | 0.372 | -0.031 [-0.127, +0.064] | -0.7 | 4.99e-01 | 7.83e-04 |
| GPT-5.6 Terra | 0.250 | 0.640 | -0.390 [-0.478, -0.303] | -9.4 | 2.30e-08 | 0.00e+00 |
| Claude Sonnet 5 | 0.548 | 0.417 | +0.131 [+0.054, +0.208] | 3.6 | 2.10e-03 | 5.46e-38 |

The unpaired two-proportion $z$ is reported only for comparison: it treats the two sides as independent samples, which they are not (the same cue $\times$ issue cells generate both), so it overstates precision. The paired issue-clustered interval is the one to quote.

### Verdict on the claim that writing hedges neutral more than prediction does

- **supports** the claim (3): Llama-3.1-8B (+0.21), Gemma-3-12B (+0.15), Claude Sonnet 5 (+0.13)
- **no difference** (1): Qwen-3.6-27B (-0.03)
- **reverses** the claim (1): GPT-5.6 Terra (-0.39)

The claim as currently written in §4.4 ("the predictions are polarised while the writing hedges to neutral far more often") is therefore **not supportable as a blanket statement across models** and must be scoped to the models where it holds. Note in particular that a model can show $\beta < 1$ (under-writing in magnitude) without hedging its writing more than its prediction: the magnitude shortfall and the neutral-composition shift are separate phenomena, and only the former is general here.

