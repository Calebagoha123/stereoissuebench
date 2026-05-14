"""Shared defaults for the thesis framing pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
STEREOISSUEBENCH_ROOT = PIPELINE_DIR.parent
WORKSPACE_ROOT = STEREOISSUEBENCH_ROOT.parent
SYSTEM_DATA_ROOT = Path("/data/kell8360/thesis_framing_pipeline")


def default_data_root() -> Path:
    env_root = os.environ.get("THESIS_PIPELINE_DATA_ROOT")
    if env_root:
        return Path(env_root)
    if SYSTEM_DATA_ROOT.exists():
        return SYSTEM_DATA_ROOT
    return PIPELINE_DIR


DATA_ROOT = default_data_root()
INPUT_DATA_DIR = DATA_ROOT / "input_data"

DEFAULT_ISSUES_CSV = INPUT_DATA_DIR / "issues_experiment.csv"
DEFAULT_TEMPLATES_CSV = INPUT_DATA_DIR / "templates_selected.csv"
DEFAULT_WORDING_CSV = PIPELINE_DIR / "issue_prompt_wording.csv"
DEFAULT_RESULTS_DIR = DATA_ROOT / "results"

DEFAULT_GEN_MODEL = "/data/resource/huggingface/models--Qwen--Qwen3.5-9B"
DEFAULT_JUDGE_MODEL = "/data/resource/huggingface/models--Qwen--Qwen3.5-4B"

DEFAULT_TEMPLATE_COUNT = 30
DEFAULT_REPEATS = 3
EXPECTED_MAIN_ISSUES = 17
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
