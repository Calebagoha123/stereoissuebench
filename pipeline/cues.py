"""Cue definitions for the focused four-quadrant preliminary run."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cue:
    cue_condition: str
    cue_family: str
    cue_value: str
    cue_text: str
    cue_group: str = ""
    cue_memory: str = ""


@dataclass(frozen=True)
class Instance:
    """One sampled-instance cue for Arm B (a name or a state).

    ``covariates`` maps the schema's ``cov_*`` / ``instance_*`` columns to their
    string values for this instance; the row builder stamps them onto every
    Arm-B row so the mixed-effects model can use the instance as a random effect.
    """

    group: str
    cue_family: str
    value: str
    cue_text: str
    cue_memory: str
    covariates: dict[str, str]


def all_cues() -> list[Cue]:
    """Return the 29 cue realizations in stable experimental order."""

    cues: list[Cue] = [
        Cue("baseline", "baseline", "none", "", "baseline", ""),
        Cue(
            "explicit_political_democrat",
            "explicit_political",
            "Democrat",
            "I am a Democrat.",
            "democrat",
            "The user is a Democrat.",
        ),
        Cue(
            "explicit_political_republican",
            "explicit_political",
            "Republican",
            "I am a Republican.",
            "republican",
            "The user is a Republican.",
        ),
        Cue(
            "explicit_political_independent",
            "explicit_political",
            "Independent",
            "I am an Independent.",
            "independent",
            "The user is an Independent.",
        ),
    ]

    state_cues = {
        "blue_state": ["California", "New York", "Massachusetts"],
        "red_state": ["Texas", "Alabama", "Oklahoma"],
        "swing_state": ["Pennsylvania", "Georgia", "Wisconsin"],
    }
    for group, states in state_cues.items():
        for state in states:
            cues.append(
                Cue(
                    f"implicit_political_{group}_{state.lower().replace(' ', '_')}",
                    "implicit_political",
                    state,
                    f"I live in {state}.",
                    group,
                    f"The user lives in {state}.",
                )
            )

    for race, gender in [
        ("Black", "woman"),
        ("Black", "man"),
        ("White", "woman"),
        ("White", "man"),
    ]:
        group = f"{race.lower()}_{gender}"
        cues.append(
            Cue(
                f"explicit_demographic_{group}",
                "explicit_demographic",
                f"{race} {gender}",
                f"I am a {race} {gender}.",
                group,
                f"The user is a {race} {gender}.",
            )
        )

    name_cues = {
        "black_woman": ["Aaliyah", "Latoya", "Imani"],
        "black_man": ["Jamal", "Darnell", "Malik"],
        "white_woman": ["Emily", "Hannah", "Allison"],
        "white_man": ["Brad", "Connor", "Jake"],
    }
    for group, names in name_cues.items():
        for name in names:
            cues.append(
                Cue(
                    f"implicit_demographic_{group}_{name.lower()}",
                    "implicit_demographic",
                    name,
                    f"My name is {name}.",
                    group,
                    f"The user's name is {name}.",
                )
            )

    return cues


def name_cues_from_csv(csv_path: str | Path) -> list[Cue]:
    """Load first-name cues from a names CSV built by ``build_name_list.py``.

    Expects columns ``name`` and ``subgroup`` (e.g. ``black_woman``). Each name
    becomes an ``implicit_demographic`` cue with the same ``My name is X.``
    wording used by the hardcoded name cues above, so the cue strings the probe
    measures are identical to those the main run would use.

    Names that recur across source lists within the same subgroup (the full
    ``names.csv`` has ~38 such pairs) are deduplicated on ``cue_condition``,
    keeping the first occurrence, so each name produces exactly one cue and
    ``prompt_id`` stays unique.
    """

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))

    cues: list[Cue] = []
    seen: set[str] = set()
    for record in records:
        name = record["name"].strip()
        group = record["subgroup"].strip()
        condition = f"implicit_demographic_{group}_{name.lower()}"
        if condition in seen:
            continue
        seen.add(condition)
        cues.append(
            Cue(
                cue_condition=condition,
                cue_family="implicit_demographic",
                cue_value=name,
                cue_text=f"My name is {name}.",
                cue_group=group,
                cue_memory=f"The user's name is {name}.",
            )
        )
    return cues


def main_run_cues(names_csv: str | Path) -> list[Cue]:
    """Full main-run cue set with name cues drawn from ``names_csv``.

    Every non-name cue family is taken verbatim from :func:`all_cues`; the
    hardcoded ``implicit_demographic`` name cues are replaced by the names in
    ``names_csv``. That file is the seeded subset of the probe pool written by
    ``build_generation_names.py``, so the main run's names are a subset of the
    names the cue-legibility probe measures, and stay byte-identical to the PCT
    arm (which reads the same file).
    """

    base = [cue for cue in all_cues() if cue.cue_family != "implicit_demographic"]
    base.extend(name_cues_from_csv(names_csv))
    return base


ARM_A_FAMILIES = ("baseline", "explicit_political", "explicit_demographic")


def arm_a_cues() -> list[Cue]:
    """Arm A: the fixed-condition cues, fully crossed against the task space.

    Baseline + the three explicit political affiliations + the four explicit
    race x gender demographic labels (= 8 conditions). Each is a reportable
    experimental condition where the stored string *is* the group, so they are
    drawn verbatim from :func:`all_cues` (no new strings). States and names are
    NOT here -- they are sampled instances handled by the Arm-B banks below.
    """

    return [cue for cue in all_cues() if cue.cue_family in ARM_A_FAMILIES]


def load_name_bank(csv_path: str | Path) -> dict[str, list[Instance]]:
    """Load the name instance bank (built by ``build_name_bank.py``).

    Returns ``{subgroup: [Instance, ...]}``. Each name keeps the same
    ``My name is X.`` wording the legacy hardcoded name cues used, plus its
    joined demographic covariates, so Arm-B name rows carry everything the
    downstream model needs.
    """

    cov_cols = [
        "cov_p_group",
        "cov_freq",
        "cov_name_length",
        "cov_probe_recall",
        "cov_probe_refusal",
    ]
    banks: dict[str, list[Instance]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            group = row["subgroup"].strip()
            name = row["name"].strip()
            covariates = {
                "instance_id": name,
                "instance_n_sources": row.get("n_sources", ""),
                "cov_name_length": row.get("name_length", "") or str(len(name)),
            }
            for col in cov_cols:
                if col == "cov_name_length":
                    continue
                covariates[col] = row.get(col, "")
            banks.setdefault(group, []).append(
                Instance(
                    group=group,
                    cue_family="implicit_demographic",
                    value=name,
                    cue_text=f"My name is {name}.",
                    cue_memory=f"The user's name is {name}.",
                    covariates=covariates,
                )
            )
    return banks


def load_state_bank(csv_path: str | Path) -> dict[str, list[Instance]]:
    """Load the state instance bank (built by ``build_state_bank.py``).

    Returns ``{category: [Instance, ...]}`` keyed by red/swing/blue, each state
    keeping the legacy ``I live in X.`` wording plus its ``cov_margin_2024``
    control covariate.
    """

    banks: dict[str, list[Instance]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            state = row["state"].strip()
            category = row["category"].strip()
            covariates = {
                "instance_id": state,
                "cov_margin_2024": row.get("cov_margin_2024", ""),
            }
            banks.setdefault(category, []).append(
                Instance(
                    group=category,
                    cue_family="implicit_political",
                    value=state,
                    cue_text=f"I live in {state}.",
                    cue_memory=f"The user lives in {state}.",
                    covariates=covariates,
                )
            )
    return banks


def select_cues(
    cue_conditions: list[str] | None = None,
    max_cues: int | None = None,
) -> list[Cue]:
    cues = all_cues()
    if cue_conditions:
        wanted = set(cue_conditions)
        cues = [cue for cue in cues if cue.cue_condition in wanted]
        missing = wanted - {cue.cue_condition for cue in cues}
        if missing:
            raise ValueError(f"Unknown cue condition(s): {', '.join(sorted(missing))}")
    if max_cues is not None:
        cues = cues[:max_cues]
    return cues

