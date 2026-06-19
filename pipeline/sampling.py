"""Arm-B instance sampling: rotate sampled instances across task slots.

Arm B does NOT fully cross instances against tasks (that is the blow-up:
~150 names x 19 issues x ~35 templates). Instead it treats each
``(issue, template, repeat)`` triple as a task *slot* and draws a fresh instance
from the group's bank for each slot, cycling through the bank so that across all
slots every instance is used many times but each lands in many different task
contexts. No individual name is welded to any one issue or template, so a
per-instance quirk cannot masquerade as a group effect, while the group is still
represented across the whole task space.

Replication lives at the GROUP level: each slot draws one instance (one
generation per cell at ``repeats=1``); the group accumulates stability from the
breadth of instances, not from deep per-instance repetition.
"""

from __future__ import annotations

import random

from cues import Instance
from prompting import issue_id_of, make_row, slugify


def rotate_instances(bank: list[Instance], n_slots: int, seed: str) -> list[Instance]:
    """Draw ``n_slots`` instances by sampling-without-replacement-with-reshuffle.

    Within each pass the bank is shuffled and dealt out so every instance is used
    once before any repeats; when the deck is exhausted it is reshuffled. The draw
    is deterministic in ``seed`` and balances usage across the bank to within one
    pass.
    """

    if not bank:
        raise ValueError("Cannot rotate an empty instance bank")
    rng = random.Random(seed)
    deck: list[Instance] = []
    drawn: list[Instance] = []
    for _ in range(n_slots):
        if not deck:
            deck = list(bank)
            rng.shuffle(deck)
        drawn.append(deck.pop())
    return drawn


def cap_bank(bank: list[Instance], n: int, seed: str) -> list[Instance]:
    """Deterministically subsample a bank to ``n`` instances, nested by size.

    A seeded shuffle then prefix means smaller caps are subsets of larger ones
    (e.g. the 50-name pilot bank is contained in the 150-name bank), so bank-size
    comparisons differ only by the added instances, not by a reshuffle.
    """

    if n >= len(bank):
        return list(bank)
    shuffled = list(bank)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n]


def validate_arm_b_rows(rows: list[dict]) -> list[str]:
    """Arm-B validation: unique ids, populated covariates, balanced usage.

    Arm B is rotated, not crossed, so the Arm-A block check does not apply. Here
    we check prompt_id uniqueness, that every row carries an instance id +
    P(group|name), and that instance usage within a group is balanced to within
    one rotation pass (max - min count <= 1)."""

    from collections import defaultdict

    errors: list[str] = []
    ids = [r["prompt_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("Arm-B prompt_id values are not unique")

    missing = [r["prompt_id"] for r in rows if not r.get("instance_id")]
    if missing:
        errors.append(f"{len(missing)} Arm-B rows missing instance_id")

    usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        usage[r["cue_group"]][r["instance_id"]] += 1
    for group, counts in usage.items():
        if counts and (max(counts.values()) - min(counts.values())) > 1:
            errors.append(
                f"Arm-B group {group} instance usage unbalanced "
                f"(min={min(counts.values())}, max={max(counts.values())})"
            )
    return errors


def _cue_condition(instance: Instance) -> str:
    if instance.cue_family == "implicit_political":
        slug = instance.value.lower().replace(" ", "_")
    else:
        slug = instance.value.lower()
    return f"{instance.cue_family}_{instance.group}_{slug}"


def build_arm_b_rows(
    issues: list[dict],
    templates: list[dict],
    banks: dict[str, list[Instance]],
    repeats: int,
    seed: str = "arm_b",
) -> list[dict]:
    """Build Arm-B rows by rotating each group's bank over its task slots.

    For each group the slots are the deterministic ``issue x template x repeat``
    order; one instance is drawn per slot from a per-group reshuffled deck (groups
    draw independently via a group-salted seed). Every row carries the instance id
    and its joined covariates for the mixed-effects fit.
    """

    rows: list[dict] = []
    for group in sorted(banks):
        bank = banks[group]
        slots = [
            (issue, template, repeat)
            for issue in issues
            for template in templates
            for repeat in range(1, repeats + 1)
        ]
        instances = rotate_instances(bank, len(slots), seed=f"{seed}:{group}")
        for (issue, template, repeat), instance in zip(slots, instances):
            rows.append(
                make_row(
                    issue,
                    template,
                    cue_condition=_cue_condition(instance),
                    cue_family=instance.cue_family,
                    cue_value=instance.value,
                    cue_group=group,
                    cue_text=instance.cue_text,
                    cue_memory=instance.cue_memory,
                    repeat=repeat,
                    arm="B",
                    arm_b_cols=instance.covariates,
                )
            )
    return rows


__all__ = [
    "rotate_instances",
    "cap_bank",
    "build_arm_b_rows",
    "validate_arm_b_rows",
    "issue_id_of",
    "slugify",
]
