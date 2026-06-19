"""Shared defaults for the thesis framing pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent

# Committed experiment inputs live in the repo and are read directly.
INPUT_DIR = REPO_ROOT / "data" / "input"
REFERENCE_DIR = REPO_ROOT / "data" / "reference"

# Large generated outputs go to a scratch/output root. On the VM this is a fast
# data volume; locally it falls back to the repo's gitignored data/processed.
SYSTEM_DATA_ROOT = Path("/data/kell8360/thesis_framing_pipeline")


def default_output_root() -> Path:
    env_root = os.environ.get("THESIS_PIPELINE_DATA_ROOT")
    if env_root:
        return Path(env_root)
    if SYSTEM_DATA_ROOT.exists():
        return SYSTEM_DATA_ROOT
    return REPO_ROOT / "data" / "processed"


OUTPUT_ROOT = default_output_root()

DEFAULT_ISSUES_CSV = INPUT_DIR / "issues_experiment.csv"
DEFAULT_TEMPLATES_CSV = INPUT_DIR / "templates_run_30.csv"
# Arm A is fully crossed against the full 145-template pool; Arm B reduces this
# same pool to a genre-preserving subset (~35) so the instance dimension stays
# tractable. Both arms read from this file.
DEFAULT_TEMPLATES_ALL_CSV = INPUT_DIR / "templates_all_145.csv"
DEFAULT_NAME_BANK_CSV = INPUT_DIR / "names" / "name_bank.csv"
DEFAULT_STATE_BANK_CSV = INPUT_DIR / "states" / "state_bank.csv"
DEFAULT_WORDING_CSV = INPUT_DIR / "issue_prompt_wording.csv"
DEFAULT_RESULTS_DIR = OUTPUT_ROOT / "results"

# Political Compass Test arm (pipeline/06_run_pct.py). The 62-item instrument is
# the verbatim coded list from Rozado/cssmodels (biasissycophancy); the name cue
# set is the seeded probe-pool subset in names_generation.csv (built by
# build_generation_names.py) -- the same names the main generation run uses, so
# the cue strings are byte-identical across arms.
DEFAULT_PCT_CSV = INPUT_DIR / "pct" / "pct_items_coded.csv"
DEFAULT_NAMES_GEN_CSV = INPUT_DIR / "names" / "names_generation.csv"

DEFAULT_GEN_MODEL = "/data/resource/huggingface/models--Qwen--Qwen3.5-9B"
DEFAULT_JUDGE_MODEL = "/data/resource/huggingface/models--Qwen--Qwen3.5-4B"

DEFAULT_TEMPLATE_COUNT = 30
DEFAULT_REPEATS = 3
EXPECTED_MAIN_ISSUES = 19
EXPECTED_CUE_REALIZATIONS = 29
EXPECTED_PILOT_ROWS = (
    EXPECTED_MAIN_ISSUES
    * DEFAULT_TEMPLATE_COUNT
    * EXPECTED_CUE_REALIZATIONS
    * DEFAULT_REPEATS
)

PROMPT_COLUMNS = [
    "prompt_id",
    "issue_id",
    "ces_variable",
    "issue_cluster",
    "template_id",
    "template_rank",
    "template_text",
    "cue_condition",
    "cue_family",
    "cue_value",
    "cue_text",
    "cue_group",
    "cue_memory",
    "generation_repeat",
    "seed",
    "prompt_topic",
    "prompt_topic_support",
    "prompt_topic_oppose",
    "system_text",
    "prompt_text",
    "topic_neutral",
    "topic_support",
    "topic_oppose",
    "stance_target",
    "liberal_sign",
    # Two-arm sampling design. Arm A (fixed-condition cues) leaves the
    # instance/covariate columns blank; Arm B (sampled-instance cues: names,
    # states) populates them so the downstream mixed-effects model can treat the
    # instance as a random effect with linguistic covariates as controls.
    "arm",
    "instance_id",
    "instance_n_sources",
    "cov_p_group",
    "cov_freq",
    "cov_name_length",
    "cov_probe_recall",
    "cov_probe_refusal",
    "cov_margin_2024",
]

# Columns Arm B fills from the instance bank (blank for Arm A). Kept as a named
# group so the row builders stamp a consistent blank template on every row.
ARM_B_COLUMNS = [
    "instance_id",
    "instance_n_sources",
    "cov_p_group",
    "cov_freq",
    "cov_name_length",
    "cov_probe_recall",
    "cov_probe_refusal",
    "cov_margin_2024",
]

GENERATION_COLUMNS = PROMPT_COLUMNS + [
    "generation_model",
    "response_text",
    "finish_reason",
]

EVAL_COLUMNS = GENERATION_COLUMNS + [
    "judge_model",
    "eval_text",
    "eval_label",
    "collapsed_stance",
    "support_score",
    "liberal_score",
]

PCT_COLUMNS = [
    "prompt_id",
    "pct_id",
    "axis",
    "ideo_direction",
    "direction_label",
    "cue_condition",
    "cue_family",
    "cue_value",
    "cue_group",
    "cue_text",
    "preamble_style",
    "repeat",
    "seed",
    "statement",
    "prompt_text",
    "pct_model",
    "response_text",
    "finish_reason",
    "letter",
    "agree_score",
    "liberal_score",
]
