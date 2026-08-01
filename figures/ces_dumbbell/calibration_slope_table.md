| Model | β, all cues | β, party labels removed |
|---|:--:|:--:|
| Qwen-3.6-27B | 1.16 [0.23, 1.61] | 0.33 [0.09, 0.57] |
| Gemma-3-12B | 1.27 [0.46, 1.60] | 0.65 [0.14, 1.21] |
| Llama-3.1-8B | 0.49 [0.17, 0.66] | 0.32 [-0.04, 0.72] |
| GPT-5.6 Terra | 0.79 [0.16, 1.18] | 0.30 [0.14, 0.50] |
| Claude Sonnet 5 | 0.88 [0.35, 1.18] | 0.49 [0.22, 0.80] |
| All models (pooled) | 0.95 [0.31, 1.30] | 0.43 [0.13, 0.80] |

Deming (errors-in-variables) calibration slope; 95% CIs are cue-clustered bootstrap. NOTE: the all-cues column is high-leverage (Democrat + Republican carry 42.5% of it) and its CIs are too wide to decide anything; the party-removed column is the informative one. β = 1 is perfect calibration; β < 1 means the model reproduces only that fraction of the real CES subgroup gap (flattening). The gap between the two columns is the share of calibration carried by the three party-label cues — pooled, the slope falls 0.95 → 0.43 once they are removed.
