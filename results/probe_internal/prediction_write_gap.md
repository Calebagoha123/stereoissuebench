## Predicted vs. written: inference at the issue level

Clustered bootstrap over issues: 10000 draws, seed 20260730, 19 issue clusters. Model enters each slope as a fixed effect, so beta is a within-model transmission rate.

`beta = 1` is the reference: a unit of predicted shift becoming a unit of written shift. BH q-values are over this table's family of 7 tests.

| Cue type | obs | mean \|predicted\| | mean \|written\| | β [95% CI] | β≠1? | p | BH q |
|---|--:|--:|--:|--:|:--:|--:|--:|
| Party label | 285 | 0.475 [0.441, 0.510] | 0.300 [0.250, 0.349] | 0.57 [0.48, 0.65] | yes | <0.0002 | <0.0002 |
| Race x gender | 380 | 0.292 [0.250, 0.334] | 0.097 [0.073, 0.121] | 0.21 [0.18, 0.24] | yes | <0.0002 | <0.0002 |
| State | 285 | 0.332 [0.304, 0.359] | 0.080 [0.063, 0.098] | 0.07 [0.05, 0.10] | yes | <0.0002 | <0.0002 |
| Name | 380 | 0.182 [0.153, 0.211] | 0.061 [0.051, 0.071] | 0.04 [-0.03, 0.10] | yes | <0.0002 | <0.0002 |

### Between-cue-type slope contrasts (the transmission gradient)

| Contrast | Δβ [95% CI] | p | BH q |
|---|--:|--:|--:|
| Party label − State | 0.50 [0.42, 0.57] | <0.0002 | <0.0002 |
| Race x gender − State | 0.14 [0.12, 0.17] | <0.0002 | <0.0002 |
| State − Name | 0.03 [-0.02, 0.09] | 0.2462 | 0.2462 |
