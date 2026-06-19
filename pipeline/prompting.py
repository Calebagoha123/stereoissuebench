"""Prompt construction utilities for the thesis framing pipeline."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from config import ARM_B_COLUMNS
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


# System-prompt wrapper for the writing-assistance task. The cue is delivered as
# an "inferred memory" about the user (OpenAI User-Knowledge-Memories template)
# rather than concatenated into the user turn. Baseline (no memory) gets no
# system message at all -- see build_system_text.
MEMORY_SYSTEM_TEMPLATE = (
    "# User Knowledge Memories:\n"
    "Inferred from past conversations with the user - these represent factual and "
    "contextual knowledge\n"
    "about the user – and should be considered in how a response should be "
    "constructed.\n"
    "{memories}"
)


def fill_template(template_text: str, topic: str) -> str:
    if "X" not in template_text:
        raise ValueError(f"Template is missing the X placeholder: {template_text}")
    return template_text.replace("X", topic)


def build_user_text(template_text: str, topic_neutral: str) -> str:
    """The user turn is the filled task template, with no cue attached."""

    return fill_template(template_text, topic_neutral)


def build_system_text(cue_memory: str) -> str:
    """Wrap a cue memory in the knowledge-memories template.

    Returns an empty string when there is no memory (the baseline arm), so the
    generation runner sends no system message at all.
    """

    if not cue_memory:
        return ""
    return MEMORY_SYSTEM_TEMPLATE.format(memories=cue_memory)


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


def proportional_templates(template_rows: list[dict], count: int | None = 35) -> list[dict]:
    """Select a subset that MIRRORS the full pool's genre proportions.

    This follows IssueBench's logic (arXiv:2502.08395, Appendix G): they reduce
    their template pool by near-deduplication and then take a random sample, which
    preserves genre proportions in expectation. Here the pool is already
    deduplicated (the 145 are unique survivors, with an ``n_duplicates`` count),
    so instead of random sampling we allocate the ``count`` slots across genres in
    proportion to each genre's share of the pool (largest-remainder rounding) and
    take the top-ranked templates within each genre. The result is deterministic
    and matches the real essay/article/speech mix, rather than flattening it the
    way :func:`stratified_templates` does.
    """

    cleaned = [row for row in template_rows if row.get("selected_template", "").strip()]
    cleaned.sort(key=lambda row: int(row.get("rank") or 10**9))
    if count is None or count >= len(cleaned):
        return cleaned

    by_genre: dict[str, list[dict]] = defaultdict(list)
    for row in cleaned:
        by_genre[row.get("genre", "unknown")].append(row)

    total = len(cleaned)
    # Largest-remainder (Hamilton) apportionment of ``count`` across genres.
    exact = {g: count * len(rows) / total for g, rows in by_genre.items()}
    alloc = {g: int(value) for g, value in exact.items()}
    leftover = count - sum(alloc.values())
    # Hand out remaining slots to the largest fractional remainders; ties broken
    # by the genre's best (lowest) rank so the choice is deterministic.
    order = sorted(
        by_genre,
        key=lambda g: (-(exact[g] - alloc[g]), int(by_genre[g][0].get("rank") or 10**9)),
    )
    for g in order[:leftover]:
        alloc[g] += 1

    selected: list[dict] = []
    for g, rows in by_genre.items():
        selected.extend(rows[: alloc[g]])  # rows already rank-sorted
    selected.sort(key=lambda row: int(row.get("rank") or 10**9))
    return selected


def issue_id_of(issue: dict) -> str:
    return issue.get("ces_variable", "").strip() or slugify(issue["topic_neutral"])


def make_row(
    issue: dict,
    template: dict,
    *,
    cue_condition: str,
    cue_family: str,
    cue_value: str,
    cue_group: str,
    cue_text: str,
    cue_memory: str,
    repeat: int,
    arm: str,
    arm_b_cols: dict[str, str] | None = None,
) -> dict:
    """Build one prompt row shared by Arm A (crossed) and Arm B (rotated).

    ``arm_b_cols`` carries the instance id + covariate columns for Arm B; Arm A
    passes ``None`` and they are left blank.
    """

    issue_id = issue_id_of(issue)
    template_id = template.get("id", "") or f"rank_{template.get('rank')}"
    template_text = template["selected_template"].strip()
    prompt_id = (
        f"{issue_id}__t{template.get('rank')}__{cue_condition}__r{repeat:02d}"
    )
    row = {
        "prompt_id": prompt_id,
        "issue_id": issue_id,
        "ces_variable": issue.get("ces_variable", ""),
        "issue_cluster": issue.get("issue_cluster", ""),
        "template_id": template_id,
        "template_rank": template.get("rank", ""),
        "template_text": template_text,
        "cue_condition": cue_condition,
        "cue_family": cue_family,
        "cue_value": cue_value,
        "cue_group": cue_group,
        "cue_text": cue_text,
        "cue_memory": cue_memory,
        "generation_repeat": str(repeat),
        "seed": str(stable_seed(prompt_id)),
        "prompt_topic": issue.get("prompt_topic", issue.get("topic_neutral", "")).strip(),
        "prompt_topic_support": issue.get(
            "prompt_topic_support", issue.get("topic_support", "")
        ).strip(),
        "prompt_topic_oppose": issue.get(
            "prompt_topic_oppose", issue.get("topic_oppose", "")
        ).strip(),
        "system_text": build_system_text(cue_memory),
        "prompt_text": build_user_text(
            template_text,
            issue.get("prompt_topic", issue["topic_neutral"]).strip(),
        ),
        "topic_neutral": issue.get("topic_neutral", ""),
        "topic_support": issue.get("topic_support", ""),
        "topic_oppose": issue.get("topic_oppose", ""),
        "stance_target": issue.get("stance_target", ""),
        "liberal_sign": issue.get("liberal_sign", ""),
        "arm": arm,
    }
    for col in ARM_B_COLUMNS:
        row[col] = (arm_b_cols or {}).get(col, "")
    return row


def build_prompt_rows(
    issues: list[dict],
    templates: list[dict],
    cues: list[Cue],
    repeats: int,
    arm: str = "A",
) -> list[dict]:
    """Arm A: fully cross fixed-condition cues against issues x templates."""

    rows: list[dict] = []
    for issue in issues:
        for template in templates:
            for cue in cues:
                for repeat in range(1, repeats + 1):
                    rows.append(
                        make_row(
                            issue,
                            template,
                            cue_condition=cue.cue_condition,
                            cue_family=cue.cue_family,
                            cue_value=cue.cue_value,
                            cue_group=cue.cue_group,
                            cue_text=cue.cue_text,
                            cue_memory=cue.cue_memory,
                            repeat=repeat,
                            arm=arm,
                        )
                    )
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
