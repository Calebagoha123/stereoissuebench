#!/usr/bin/env python3
"""Validate thesis input files and expected preliminary-run counts."""

from __future__ import annotations

import argparse
import sys

from config import (
    DEFAULT_ISSUES_CSV,
    DEFAULT_TEMPLATE_COUNT,
    DEFAULT_TEMPLATES_CSV,
    DEFAULT_WORDING_CSV,
    EXPECTED_MAIN_ISSUES,
    EXPECTED_PILOT_ROWS,
    EXPECTED_CUE_REALIZATIONS,
    DEFAULT_REPEATS,
)
from cues import all_cues, select_cues
from io_utils import read_csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issues", default=str(DEFAULT_ISSUES_CSV))
    parser.add_argument("--templates", default=str(DEFAULT_TEMPLATES_CSV))
    parser.add_argument("--wording", default=str(DEFAULT_WORDING_CSV))
    parser.add_argument("--template-count", type=int, default=DEFAULT_TEMPLATE_COUNT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--issue-limit", type=int)
    parser.add_argument("--max-cues", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issue_rows = read_csv(args.issues)
    template_rows = read_csv(args.templates)
    wording_rows = read_csv(args.wording)

    errors: list[str] = []
    errors.extend(validate_columns(issue_rows, REQUIRED_ISSUE_COLUMNS, "issues CSV"))
    errors.extend(validate_columns(template_rows, REQUIRED_TEMPLATE_COLUMNS, "templates CSV"))
    errors.extend(validate_columns(wording_rows, REQUIRED_WORDING_COLUMNS, "wording CSV"))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    issue_rows = apply_issue_wording(issue_rows, wording_rows)
    issues = main_issues(issue_rows, args.issue_limit)
    templates = stratified_templates(template_rows, args.template_count)
    cues = select_cues(max_cues=args.max_cues)
    prompt_rows = build_prompt_rows(issues, templates, cues, args.repeats)
    errors.extend(validate_prompt_rows(prompt_rows, len(cues)))

    wording_ids = {row["ces_variable"] for row in wording_rows}
    missing_wording = [
        row["ces_variable"]
        for row in main_issues(issue_rows, args.issue_limit)
        if row.get("ces_variable") not in wording_ids
    ]
    if missing_wording:
        errors.append(
            "main issues missing open-direction prompt wording: "
            + ", ".join(sorted(missing_wording))
        )

    missing_template_placeholders = [
        row.get("id", row.get("rank", "unknown"))
        for row in templates
        if "X" not in row.get("selected_template", "")
    ]
    if missing_template_placeholders:
        errors.append(
            f"{len(missing_template_placeholders)} selected templates are missing the X placeholder"
        )

    for row in issues:
        if row.get("liberal_sign") not in {"1", "-1"}:
            errors.append(
                f"{row.get('ces_variable', row.get('topic_neutral'))} has invalid liberal_sign"
            )

    if len(all_cues()) != EXPECTED_CUE_REALIZATIONS:
        errors.append(f"cue expansion produced {len(all_cues())} cues, expected 29")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    expected_rows = len(issues) * len(templates) * len(cues) * args.repeats
    print("Validation passed.")
    print(f"Main issues: {len(issues)}")
    print(f"Templates: {len(templates)}")
    print(f"Cue realizations: {len(cues)}")
    print(f"Repeats: {args.repeats}")
    print(f"Prompt rows: {expected_rows}")
    if (
        len(issues) == EXPECTED_MAIN_ISSUES
        and len(templates) == DEFAULT_TEMPLATE_COUNT
        and len(cues) == EXPECTED_CUE_REALIZATIONS
        and args.repeats == DEFAULT_REPEATS
    ):
        print(f"Pilot expected rows confirmed: {EXPECTED_PILOT_ROWS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
