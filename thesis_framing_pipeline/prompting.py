"""Prompt construction utilities for the thesis framing pipeline."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from cues import Cue


REQUIRED_ISSUE_COLUMNS = {
    "analysis_tier",
    "issue_cluster",
    "ces_variable",
    "topic_neutral",
    "topic_support",
    "topic_oppose",
    "stance_target",
    "liberal_sign",
}

REQUIRED_TEMPLATE_COLUMNS = {
    "rank",
    "id",
    "selected_template",
    "genre",
}

REQUIRED_WORDING_COLUMNS = {
    "ces_variable",
    "prompt_topic",
    "prompt_topic_support",
    "prompt_topic_oppose",
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def fill_template(template_text: str, topic: str) -> str:
    if "X" not in template_text:
        raise ValueError(f"Template is missing the X placeholder: {template_text}")
    return template_text.replace("X", topic)


def build_prompt_text(cue_text: str, template_text: str, topic_neutral: str) -> str:
    filled = fill_template(template_text, topic_neutral)
    if cue_text:
        return f"{cue_text}\n\n{filled}"
    return filled


def apply_issue_wording(issue_rows: list[dict], wording_rows: list[dict]) -> list[dict]:
    """Attach open-direction prompt wording keyed by CES variable."""

    wording_by_issue = {row["ces_variable"]: row for row in wording_rows}
    merged: list[dict] = []
    for issue in issue_rows:
        row = dict(issue)
        wording = wording_by_issue.get(row.get("ces_variable", ""))
        if wording:
            row["prompt_topic"] = wording["prompt_topic"].strip()
            row["prompt_topic_support"] = wording["prompt_topic_support"].strip()
            row["prompt_topic_oppose"] = wording["prompt_topic_oppose"].strip()
        else:
            row["prompt_topic"] = row.get("topic_neutral", "").strip()
            row["prompt_topic_support"] = row.get("topic_support", "").strip()
            row["prompt_topic_oppose"] = row.get("topic_oppose", "").strip()
        merged.append(row)
    return merged


def validate_columns(rows: list[dict], required: set[str], label: str) -> list[str]:
    if not rows:
        return [f"{label} is empty"]
    missing = sorted(required - set(rows[0].keys()))
    return [f"{label} missing required column: {col}" for col in missing]


def main_issues(issue_rows: list[dict], issue_limit: int | None = None) -> list[dict]:
    rows = [row for row in issue_rows if row.get("analysis_tier") == "main"]
    if issue_limit is not None:
        rows = rows[:issue_limit]
    return rows


def stratified_templates(template_rows: list[dict], count: int | None = 30) -> list[dict]:
    """Select a deterministic genre-preserving subset by round-robin over rank."""

    cleaned = [row for row in template_rows if row.get("selected_template", "").strip()]
    cleaned.sort(key=lambda row: int(row.get("rank") or 10**9))
    if count is None or count >= len(cleaned):
        return cleaned

    by_genre: dict[str, list[dict]] = defaultdict(list)
    for row in cleaned:
        by_genre[row.get("genre", "unknown")].append(row)

    selected: list[dict] = []
    genres = sorted(by_genre, key=lambda genre: int(by_genre[genre][0].get("rank") or 10**9))
    while len(selected) < count:
        progressed = False
        for genre in genres:
            if by_genre[genre]:
                selected.append(by_genre[genre].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break

    selected.sort(key=lambda row: int(row.get("rank") or 10**9))
    return selected


def build_prompt_rows(
    issues: list[dict],
    templates: list[dict],
    cues: list[Cue],
    repeats: int,
) -> list[dict]:
    rows: list[dict] = []
    for issue in issues:
        issue_id = issue.get("ces_variable", "").strip() or slugify(issue["topic_neutral"])
        for template in templates:
            template_id = template.get("id", "") or f"rank_{template.get('rank')}"
            template_text = template["selected_template"].strip()
            for cue in cues:
                for repeat in range(1, repeats + 1):
                    prompt_id = (
                        f"{issue_id}__t{template.get('rank')}__"
                        f"{cue.cue_condition}__r{repeat:02d}"
                    )
                    row = {
                        "prompt_id": prompt_id,
                        "issue_id": issue_id,
                        "ces_variable": issue.get("ces_variable", ""),
                        "issue_cluster": issue.get("issue_cluster", ""),
                        "template_id": template_id,
                        "template_rank": template.get("rank", ""),
                        "template_text": template_text,
                        "cue_condition": cue.cue_condition,
                        "cue_family": cue.cue_family,
                        "cue_value": cue.cue_value,
                        "cue_group": cue.cue_group,
                        "cue_text": cue.cue_text,
                        "generation_repeat": str(repeat),
                        "seed": str(stable_seed(prompt_id)),
                        "prompt_topic": issue.get("prompt_topic", issue.get("topic_neutral", "")).strip(),
                        "prompt_topic_support": issue.get(
                            "prompt_topic_support", issue.get("topic_support", "")
                        ).strip(),
                        "prompt_topic_oppose": issue.get(
                            "prompt_topic_oppose", issue.get("topic_oppose", "")
                        ).strip(),
                        "prompt_text": build_prompt_text(
                            cue.cue_text,
                            template_text,
                            issue.get("prompt_topic", issue["topic_neutral"]).strip(),
                        ),
                        "topic_neutral": issue.get("topic_neutral", ""),
                        "topic_support": issue.get("topic_support", ""),
                        "topic_oppose": issue.get("topic_oppose", ""),
                        "stance_target": issue.get("stance_target", ""),
                        "liberal_sign": issue.get("liberal_sign", ""),
                    }
                    rows.append(row)
    return rows


def validate_prompt_rows(rows: list[dict], cue_count: int) -> list[str]:
    errors: list[str] = []
    prompt_ids = [row["prompt_id"] for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        errors.append("prompt_id values are not unique")

    missing_labels = [
        row["prompt_id"]
        for row in rows
        if not row.get("topic_support") or not row.get("topic_oppose")
    ]
    if missing_labels:
        errors.append(f"{len(missing_labels)} prompt rows are missing support/oppose labels")

    missing_prompt_topics = [
        row["prompt_id"]
        for row in rows
        if not row.get("prompt_topic")
        or not row.get("prompt_topic_support")
        or not row.get("prompt_topic_oppose")
    ]
    if missing_prompt_topics:
        errors.append(f"{len(missing_prompt_topics)} prompt rows are missing prompt-topic wording")

    block_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["issue_id"], row["template_id"], str(row["generation_repeat"]))
        block_counts[key] += 1
    bad_blocks = [key for key, value in block_counts.items() if value != cue_count]
    if bad_blocks:
        errors.append(
            f"{len(bad_blocks)} matched issue/template/repeat blocks do not have {cue_count} cues"
        )
    return errors
