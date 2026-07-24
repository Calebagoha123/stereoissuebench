| Model | β, all cues | β, party labels removed |
|---|:--:|:--:|
| Qwen-3.6-27B | 0.91 [0.24, 1.23] | 0.32 [0.12, 0.53] |
| Gemma-3-12B | 0.94 [0.33, 1.18] | 0.47 [0.12, 0.82] |
| Llama-3.1-8B | 0.38 [0.17, 0.49] | 0.26 [0.00, 0.54] |
| GPT-5.6 Terra | 0.47 [0.13, 0.68] | 0.24 [0.10, 0.39] |
| Claude Sonnet 5 | 0.76 [0.26, 1.04] | 0.35 [0.11, 0.60] |
| All models (pooled) | 0.72 [0.25, 0.98] | 0.33 [0.11, 0.60] |

Deming (errors-in-variables) calibration slope; 95% CIs are cue-clustered bootstrap. β = 1 is perfect calibration; β < 1 means the model reproduces only that fraction of the real CES subgroup gap (flattening). The all-cues column matches each panel's header β in the figure. The gap between the two columns is the share of calibration carried by the three party-label cues — pooled, the slope falls 0.72 → 0.33 once they are removed.
