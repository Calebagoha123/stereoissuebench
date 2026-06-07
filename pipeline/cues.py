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


def all_cues() -> list[Cue]:
    """Return the 29 cue realizations in stable experimental order."""

    cues: list[Cue] = [
        Cue("baseline", "baseline", "none", "", "baseline"),
        Cue(
            "explicit_political_democrat",
            "explicit_political",
            "Democrat",
            "I am a Democrat.",
            "democrat",
        ),
        Cue(
            "explicit_political_republican",
            "explicit_political",
            "Republican",
            "I am a Republican.",
            "republican",
        ),
        Cue(
            "explicit_political_independent",
            "explicit_political",
            "Independent",
            "I am an Independent.",
            "independent",
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
                )
            )

    return cues


def name_cues_from_csv(csv_path: str | Path) -> list[Cue]:
    """Load first-name cues from a names CSV built by ``build_name_list.py``.

    Expects columns ``name`` and ``subgroup`` (e.g. ``black_woman``). Each name
    becomes an ``implicit_demographic`` cue with the same ``My name is X.``
    wording used by the hardcoded name cues above, so the cue strings the probe
    measures are identical to those the main run would use.
    """

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))

    cues: list[Cue] = []
    for record in records:
        name = record["name"].strip()
        group = record["subgroup"].strip()
        cues.append(
            Cue(
                cue_condition=f"implicit_demographic_{group}_{name.lower()}",
                cue_family="implicit_demographic",
                cue_value=name,
                cue_text=f"My name is {name}.",
                cue_group=group,
            )
        )
    return cues


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

