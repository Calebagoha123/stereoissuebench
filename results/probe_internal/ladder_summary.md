## Probe ladder — where the name cue stops mattering

For the **name** cue (implicit demographic), one row per model across the rungs. The name is decoded and represented like an explicit label (rungs 1–2), rated far less diagnostic than the race it carries (rung 3), and barely moves the written stance (rung 4) — a use/relevance gap, not a legibility one.

| Model | Decode bal-acc | Selectivity | Transfer name→label | Belief shift | Relevance: first name | Relevance: race | |Stance shift| | Mediation r | r (no Rep.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-3.6-27B | 1.00 | 0.76 | 0.93 | +0.047 | 2 | 53 | 0.023 | 0.94 | 0.34 |
| Gemma-3-12B | 1.00 | 0.76 | 0.91 | +0.178 | 1 | 19 | 0.043 | 0.70 | 0.13 |
| Llama-3.1-8B | 1.00 | 0.76 | 1.00 | +0.113 | 4 | 24 | 0.014 | 0.80 | 0.45 |

*Decode bal-acc / selectivity: 4-way group decodability of the name at the best layer vs a shuffled-label control. Transfer: a race×gender probe trained on explicit labels, tested on names (4-way chance 0.25). Relevance: self-rated 0–100 usefulness for predicting opinion. Stance shift is mean |·| over the four name groups (full_3x). Mediation r pairs the internal Dem–Rep axis shift with the written-stance shift across all 14 cue groups; the last column removes the Republican cue, which shows the correlation is a leverage point (see the B2/B3 limitation in the text) — the internal political-axis projection is confounded by a cue-presence offset and is not shipped as a figure.*
