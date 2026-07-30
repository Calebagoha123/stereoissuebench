#!/usr/bin/env bash
# Reproduce the full RQ2 robustness suite (docs/robustness_checks.md).
# Run from anywhere: bash analysis/05_robustness/run_robustness.sh
# All scripts read/write paths relative to the repo root, so we cd there first.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

echo "[1/9] model-shift table (issue-clustered bootstrap; ~70s)"
python3 analysis/lib/_common.py

echo "[2/9] calibration regression: OLS-origin / free-intercept / Deming"
python3 analysis/04_calibration/rq2_regression.py

echo "[3/9] stance-reduction sensitivity (threshold bands + directional-only)"
python3 analysis/04_calibration/rq2_stance_sensitivity.py

echo "[4/9] TOST equivalence on name/state nulls"
python3 analysis/05_robustness/tost_names.py

echo "[5/9] composition / variance-collapse flattening"
python3 analysis/05_robustness/composition_flattening.py

echo "[6/9] DiD variance propagation (CES design SE)"
python3 analysis/05_robustness/did_variance.py

echo "[7/9] refusal Manski bounds + generation variance + instance breakdown"
# refusal_bounds reads the EARLIER judge-scored run (results/full/eval_*.csv), the only
# local corpus retaining response text, so it covers the 3 open-weight models. The
# corpus-of-record complement is finish_reason_flatness (all 5 models, no text needed).
python3 analysis/05_robustness/refusal_bounds.py
python3 analysis/05_robustness/finish_reason_flatness.py
python3 analysis/05_robustness/generation_variance.py
python3 analysis/05_robustness/instance_breakdown.py

echo "[8/9] multiplicity (BH-FDR), leave-one-issue-out, permutation"
python3 analysis/05_robustness/rq2_extras.py

echo "[8b/9] interval-method agreement (cluster-t vs bootstrap)"
python3 analysis/05_robustness/ci_method_agreement.py

echo "[8c/9] cue x issue calibration under fixed- vs random-effect specifications (R)"
Rscript analysis/04_calibration/cue_issue_mixed.R

echo "[9/9] CLMM ordinal robustness (R; ~25 min for 5 models) + comparison"
python3 - <<'PY'
import sys; sys.path.insert(0, "analysis/lib")
from _common import load_model, CUE_ORDER, MODELS
keep = {"baseline"} | {f"{f}__{g}" for f, g in CUE_ORDER}
for m in MODELS:
    d = load_model(m)
    d["cue"] = d.apply(lambda r: "baseline" if r.cue_family == "baseline"
                       else f"{r.cue_family}__{r.cue_group}", axis=1)
    d = d[d["cue"].isin(keep)]
    d[["cue", "issue_id", "template_id", "instance", "arm", "y"]].to_csv(
        f"results/robustness/clmm_input_{m}.csv", index=False)
    print(m, len(d))
PY
Rscript analysis/05_robustness/clmm_robustness.R
python3 analysis/05_robustness/clmm_compare.py

echo "DONE. Classifier check: python3 analysis/02_stance_scorer/classifier_validation.py"
echo "Tables in results/robustness/, figures in figures/robustness/."
