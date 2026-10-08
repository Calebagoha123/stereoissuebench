## Interval-method agreement (cluster-$t$ vs percentile bootstrap)

- cells compared: **70** (5 models x 14 cues)
- point estimates: max absolute difference **0.0006** (the two differ only in how the interval is formed, not the estimator)
- significant under cluster-$t$: **45**
- significant under bootstrap: **45**
- **agreement on significance: 70/70 cells**
- cluster-$t$ intervals are wider by a median factor of **1.10** (range 1.06-1.15)

**Primary = cluster-$t$.** It is the more conservative of the two, so every claim made under it also holds under the bootstrap; it is the direct implementation of the Week-8 clustering rule including the degrees-of-freedom consequence; and it is what the headline figures display. The bootstrap is reported as an agreement check and is retained where a bootstrap variance is needed downstream (the Deming fit and the DiD variance propagation both consume `model_shift_var`).
