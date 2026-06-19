#!/usr/bin/env python3
"""Build matched prompt rows for the two-arm thesis run.

Arm A (fixed-condition cues: baseline + political + demographic labels) is fully
crossed against the full issue x template space. Arm B (sampled-instance cues:
names, states) rotates instances drawn from group-keyed banks across a reduced,
genre-preserving template subset, one instance per task slot, with replication at
the group level. See the plan ("two-arm cue sampling") for the rationale.

Examples:
    # Both arms with defaults
    python pipeline/01_build_prompts.py

    # Arm A only, quick smoke
    python pipeline/01_build_prompts.py --arm A --mode smoke

    # Pilot: one group at two bank sizes for the variance check
    python pipeline/01_build_prompts.py --arm B \
        --pilot-group black_man --pilot-bank-sizes 50,150

    # Legacy single-set crossing (back-compat)
    python pipeline/01_build_prompts.py --cue-condition baseline --templates ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    DEFAULT_ISSUES_CSV,
    DEFAULT_NAME_BANK_CSV,
    DEFAULT_NAMES_GEN_CSV,
    DEFAULT_RESULTS_DIR,
    DEFAULT_STATE_BANK_CSV,
    DEFAULT_TEMPLATE_COUNT,
    DEFAULT_TEMPLATES_ALL_CSV,
    DEFAULT_TEMPLATES_CSV,
    DEFAULT_WORDING_CSV,
    DEFAULT_REPEATS,
    PROMPT_COLUMNS,
)
from cues import arm_a_cues, load_name_bank, load_state_bank, main_run_cues, select_cues
from io_utils import read_csv, write_csv
from prompting import (
    REQUIRED_ISSUE_COLUMNS,
    REQUIRED_TEMPLATE_COLUMNS,
    REQUIRED_WORDING_COLUMNS,
    apply_issue_wording,
    build_prompt_rows,
    main_issues,
    proportional_templates,
    stratified_templates,
    validate_columns,
    validate_prompt_rows,
)
from sampling import build_arm_b_rows, cap_bank, validate_arm_b_rows

# Arm-B reduced template count and replication (reasoned defaults; settle with the
# pilot per the plan's honesty flags).
DEFAULT_ARM_B_TEMPLATES = 35
DEFAULT_ARM_B_REPEATS = 2

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
    parser.add_argument(
        "--arm",
        choices=["A", "B", "both"],
        default="both",
        help="Which arm(s) to build. Ignored when --cue-condition is given.",
    )
    parser.add_argument("--issues", default=str(DEFAULT_ISSUES_CSV))
    parser.add_argument(
        "--templates",
        default=str(DEFAULT_TEMPLATES_ALL_CSV),
        help="Template pool (default: the full 145). Arm A uses all; Arm B reduces it.",
    )
    parser.add_argument("--wording", default=str(DEFAULT_WORDING_CSV))
    parser.add_argument("--out", help="Output path (single arm) or base dir (both).")
    parser.add_argument("--issue-limit", type=int)
    parser.add_argument("--template-count", type=int, help="Cap Arm-A templates (default: all).")
    parser.add_argument("--repeats", type=int, help="Arm-A repeats per cell.")
    # Arm B
    parser.add_argument("--arm-b-templates", type=int, default=DEFAULT_ARM_B_TEMPLATES)
    parser.add_argument(
        "--template-sampling",
        choices=["proportional", "balanced"],
        default="proportional",
        help="Arm-B template subset: 'proportional' mirrors the pool's genre mix "
        "(IssueBench-style, default); 'balanced' equalizes genres (round-robin).",
    )
    parser.add_argument("--arm-b-repeats", type=int, default=DEFAULT_ARM_B_REPEATS)
    parser.add_argument("--names-bank", default=str(DEFAULT_NAME_BANK_CSV))
    parser.add_argument("--states-bank", default=str(DEFAULT_STATE_BANK_CSV))
    parser.add_argument(
        "--names-per-group",
        type=int,
        help="Cap each name group's bank to this many instances (default: full). "
        "Use a smaller value for the closed-model confirmatory slice.",
    )
    parser.add_argument(
        "--no-states", action="store_true", help="Build Arm B from names only (skip states)."
    )
    parser.add_argument("--seed", default="arm_b", help="Rotation seed for Arm B.")
    # Pilot variance-check
    parser.add_argument("--pilot-group", help="Build Arm B for only this one group.")
    parser.add_argument(
        "--pilot-bank-sizes",
        help="Comma-separated bank sizes for --pilot-group, e.g. 50,150.",
    )
    # Legacy single-set path
    parser.add_argument("--max-cues", type=int)
    parser.add_argument("--cue-condition", action="append", dest="cue_conditions")
    parser.add_argument(
        "--names",
        default=str(DEFAULT_NAMES_GEN_CSV),
        help="Legacy generation-names CSV for --cue-condition path.",
    )
    return parser.parse_args()


def _load_inputs(args) -> tuple[list[dict], list[dict]]:
    issue_rows = read_csv(args.issues)
    template_rows = read_csv(args.templates)
    wording_rows = read_csv(args.wording)
    errors = validate_columns(issue_rows, REQUIRED_ISSUE_COLUMNS, "issues CSV")
    errors.extend(validate_columns(template_rows, REQUIRED_TEMPLATE_COLUMNS, "templates CSV"))
    errors.extend(validate_columns(wording_rows, REQUIRED_WORDING_COLUMNS, "wording CSV"))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    issue_rows = apply_issue_wording(issue_rows, wording_rows)
    return issue_rows, template_rows


def _write(out: Path, rows: list[dict], label: str) -> None:
    write_csv(out, rows, PROMPT_COLUMNS)
    print(f"[{label}] Saved {len(rows)} rows to {out}")


def _fail(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)


def build_legacy(args, issues, templates, defaults) -> int:
    """Old single-set full-crossing path, kept for back-compat."""

    template_count = (
        args.template_count if args.template_count is not None else defaults["template_count"]
    )
    repeats = args.repeats if args.repeats is not None else defaults["repeats"]
    templates = stratified_templates(templates, template_count)
    if args.cue_conditions:
        cues = select_cues(args.cue_conditions, args.max_cues)
    else:
        cues = main_run_cues(args.names)
        if args.max_cues is not None:
            cues = cues[: args.max_cues]
    rows = build_prompt_rows(issues, templates, cues, repeats)
    errors = validate_prompt_rows(rows, len(cues))
    if errors:
        _fail(errors)
    out = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / f"prompts_{args.mode}.csv"
    _write(out, rows, "legacy")
    print(f"{len(issues)} issues * {len(templates)} templates * {len(cues)} cues * {repeats} reps")
    return 0


def build_arm_a(args, issues, template_rows, defaults, out_dir) -> int:
    template_count = args.template_count  # default None -> full 145
    if args.mode == "smoke" and template_count is None:
        template_count = defaults["template_count"]
    repeats = args.repeats if args.repeats is not None else DEFAULT_REPEATS
    if args.mode == "smoke":
        repeats = min(repeats, 1)
    templates = stratified_templates(template_rows, template_count)
    cues = arm_a_cues()
    rows = build_prompt_rows(issues, templates, cues, repeats, arm="A")
    errors = validate_prompt_rows(rows, len(cues))
    if errors:
        _fail(errors)
    out = Path(args.out) if (args.out and args.arm == "A") else out_dir / "prompts_arm_a.csv"
    _write(out, rows, "Arm A")
    print(f"  Arm A: {len(issues)} issues * {len(templates)} templates * {len(cues)} cues * {repeats} reps")
    return 0


def _load_arm_b_banks(args) -> dict:
    banks = load_name_bank(args.names_bank)
    if not args.no_states:
        banks.update(load_state_bank(args.states_bank))
    return banks


def build_arm_b(args, issues, template_rows, out_dir) -> int:
    n_templates = 2 if args.mode == "smoke" else args.arm_b_templates
    repeats = 1 if args.mode == "smoke" else args.arm_b_repeats
    select = proportional_templates if args.template_sampling == "proportional" else stratified_templates
    templates = select(template_rows, n_templates)

    # Pilot: one group, multiple bank sizes, matched and nested.
    if args.pilot_group:
        if not args.pilot_bank_sizes:
            print("ERROR: --pilot-group requires --pilot-bank-sizes", file=sys.stderr)
            return 1
        name_banks = load_name_bank(args.names_bank)
        if args.pilot_group not in name_banks:
            print(
                f"ERROR: unknown --pilot-group {args.pilot_group}; "
                f"have {sorted(name_banks)}",
                file=sys.stderr,
            )
            return 1
        full = name_banks[args.pilot_group]
        for size in [int(s) for s in args.pilot_bank_sizes.split(",")]:
            capped = cap_bank(full, size, seed=f"{args.seed}:pilot:{args.pilot_group}")
            rows = build_arm_b_rows(
                issues, templates, {args.pilot_group: capped}, repeats, seed=args.seed
            )
            errors = validate_arm_b_rows(rows)
            if errors:
                _fail(errors)
            out = out_dir / f"prompts_pilot_{args.pilot_group}_n{size}.csv"
            _write(out, rows, f"pilot n={len(capped)}")
            print(
                f"  pilot {args.pilot_group}: {len(capped)} names * {len(issues)} issues "
                f"* {len(templates)} templates * {repeats} reps = {len(rows)}"
            )
        return 0

    banks = _load_arm_b_banks(args)
    if args.names_per_group is not None:
        name_groups = set(load_name_bank(args.names_bank))
        banks = {
            g: (cap_bank(b, args.names_per_group, seed=f"{args.seed}:cap:{g}") if g in name_groups else b)
            for g, b in banks.items()
        }
    rows = build_arm_b_rows(issues, templates, banks, repeats, seed=args.seed)
    errors = validate_arm_b_rows(rows)
    if errors:
        _fail(errors)
    out = Path(args.out) if (args.out and args.arm == "B") else out_dir / "prompts_arm_b.csv"
    _write(out, rows, "Arm B")
    sizes = {g: len(b) for g, b in sorted(banks.items())}
    print(
        f"  Arm B: groups {sizes} * {len(issues)} issues * {len(templates)} templates "
        f"* {repeats} reps = {len(rows)}"
    )
    return 0


def main() -> int:
    args = parse_args()
    defaults = MODE_DEFAULTS[args.mode]
    issue_limit = args.issue_limit if args.issue_limit is not None else defaults["issue_limit"]

    issue_rows, template_rows = _load_inputs(args)
    issues = main_issues(issue_rows, issue_limit)

    # Legacy path takes precedence when explicit cue conditions are requested.
    if args.cue_conditions:
        return build_legacy(args, issues, template_rows, defaults)

    # --out is a directory for multi-output runs (both arms, or a pilot sweep)
    # and a single file for a one-arm, non-pilot build.
    dir_output = args.arm == "both" or bool(args.pilot_group)
    out_dir = Path(args.out) if (args.out and dir_output) else DEFAULT_RESULTS_DIR

    rc = 0
    if args.arm in ("A", "both"):
        rc |= build_arm_a(args, issues, template_rows, defaults, out_dir)
    if args.arm in ("B", "both"):
        rc |= build_arm_b(args, issues, template_rows, out_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
