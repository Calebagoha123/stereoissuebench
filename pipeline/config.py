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
DEFAULT_WORDING_CSV = INPUT_DIR / "issue_prompt_wording.csv"
DEFAULT_RESULTS_DIR = OUTPUT_ROOT / "results"

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
    "generation_repeat",
    "seed",
    "prompt_topic",
    "prompt_topic_support",
    "prompt_topic_oppose",
    "prompt_text",
    "topic_neutral",
    "topic_support",
    "topic_oppose",
    "stance_target",
    "liberal_sign",
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
