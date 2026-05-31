#!/usr/bin/env python3
"""Build matched prompt rows for the preliminary thesis run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    DEFAULT_ISSUES_CSV,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEMPLATE_COUNT,
    DEFAULT_TEMPLATES_CSV,
    DEFAULT_WORDING_CSV,
    DEFAULT_REPEATS,
    PROMPT_COLUMNS,
)
from cues import select_cues
from io_utils import read_csv, write_csv
from prompting import (
    REQUIRED_ISSUE_COLUMNS,
    REQUIRED_TEMPLATE_COLUMNS,
    REQUIRED_WORDING_COLUMNS,
    apply_issue_wording,
    build_prompt_rows,
    main_issues,
    stratified_templates,
    validate_columns,
    validate_prompt_rows,
)


MODE_DEFAULTS = {
    "smoke": {"issue_limit": 1, "template_count": 2, "max_cues": 3, "repeats": 1},
    "pilot": {
        "issue_limit": None,
        "template_count": DEFAULT_TEMPLATE_COUNT,
        "max_cues": None,
        "repeats": DEFAULT_REPEATS,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--issues", default=str(DEFAULT_ISSUES_CSV))
    parser.add_argument("--templates", default=str(DEFAULT_TEMPLATES_CSV))
    parser.add_argument("--wording", default=str(DEFAULT_WORDING_CSV))
    parser.add_argument("--out")
    parser.add_argument("--issue-limit", type=int)
    parser.add_argument("--template-count", type=int)
    parser.add_argument("--max-cues", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--cue-condition", action="append", dest="cue_conditions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = MODE_DEFAULTS[args.mode]
    issue_limit = args.issue_limit if args.issue_limit is not None else defaults["issue_limit"]
    template_count = (
        args.template_count if args.template_count is not None else defaults["template_count"]
    )
    max_cues = args.max_cues if args.max_cues is not None else defaults["max_cues"]
    repeats = args.repeats if args.repeats is not None else defaults["repeats"]
    out = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / f"prompts_{args.mode}.csv"

    issue_rows = read_csv(args.issues)
    template_rows = read_csv(args.templates)
    wording_rows = read_csv(args.wording)
    errors = validate_columns(issue_rows, REQUIRED_ISSUE_COLUMNS, "issues CSV")
    errors.extend(validate_columns(template_rows, REQUIRED_TEMPLATE_COLUMNS, "templates CSV"))
    errors.extend(validate_columns(wording_rows, REQUIRED_WORDING_COLUMNS, "wording CSV"))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    issue_rows = apply_issue_wording(issue_rows, wording_rows)
    issues = main_issues(issue_rows, issue_limit)
    templates = stratified_templates(template_rows, template_count)
    cues = select_cues(args.cue_conditions, max_cues)
    rows = build_prompt_rows(issues, templates, cues, repeats)
    errors = validate_prompt_rows(rows, len(cues))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    write_csv(out, rows, PROMPT_COLUMNS)
    print(f"Saved {len(rows)} prompt rows to {out}")
    print(
        f"{len(issues)} issues * {len(templates)} templates * "
        f"{len(cues)} cues * {repeats} repeats = {len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
