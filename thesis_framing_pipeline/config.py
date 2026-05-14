"""Shared defaults for the thesis framing pipeline."""

from __future__ import annotations

from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
STEREOISSUEBENCH_ROOT = PIPELINE_DIR.parent
WORKSPACE_ROOT = STEREOISSUEBENCH_ROOT.parent

DEFAULT_ISSUES_CSV = WORKSPACE_ROOT / "issues_experiment.csv"
DEFAULT_TEMPLATES_CSV = WORKSPACE_ROOT / "templates_selected.csv"
DEFAULT_WORDING_CSV = PIPELINE_DIR / "issue_prompt_wording.csv"
DEFAULT_RESULTS_DIR = PIPELINE_DIR / "results"

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
