#!/usr/bin/env python3
"""Build a race x gender first-name list for the cue-legibility probe.

Replicates the Tonneau et al. (2026, arXiv:2601.18486) Appendix A.1 recipe:
combine a name source's race-specificity scores with U.S. Social Security
Administration gender shares, then retain the most strongly associated names
per race-gender subgroup.

v1 wires the Tzioumis (2018) source. Rosenman et al. (2023) and Elder & Hayes
(2023) are added as additional `NameSource` entries once their raw files are
available under ``data/input/names/raw/``; each source is selected independently
so the per-source lists can be unioned downstream.

Usage:
    python pipeline/build_name_list.py --source tzioumis --per-cell 50
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMES_DIR = REPO_ROOT / "data" / "input" / "names"
RAW_DIR = NAMES_DIR / "raw"

# Harvard Dataverse access endpoint for Tzioumis (2018), doi:10.7910/DVN/TYJKEZ,
# firstnames.xlsx (datafile id 3078263). ?format=original returns the workbook.
TZIOUMIS_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/3078263?format=original"
)
# SSA national gender shares via a public GitHub mirror (SSA direct is firewalled
# in some environments). Long format: year, name, percent, sex in {boy, girl}.
SSA_GENDER_URL = (
    "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv"
)

# The four subgroups the probe scores, matching pipeline/cues.py.
RACES = ["White", "Black"]
GENDERS = ["man", "woman"]


@dataclass(frozen=True)
class NameSource:
    """A name source that yields per-name race-specificity scores."""

    key: str
    raw_filename: str
    download_url: str

    def race_specificity(self, race: str) -> str:
        """Column giving the % of people with this name who are `race`."""
        return {"White": "pctwhite", "Black": "pctblack"}[race]


TZIOUMIS = NameSource(
    key="tzioumis",
    raw_filename="tzioumis_firstnames.xlsx",
    download_url=TZIOUMIS_URL,
)

SOURCES = {TZIOUMIS.key: TZIOUMIS}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url}\n        -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def load_tzioumis(source: NameSource) -> pd.DataFrame:
    """Return name, obs, pctwhite, pctblack (name title-cased for matching)."""
    raw = RAW_DIR / source.raw_filename
    _download(source.download_url, raw)
    frame = pd.read_excel(raw, sheet_name="Data", engine="openpyxl")
    frame = frame.rename(columns={"firstname": "name"})
    frame["name"] = frame["name"].astype(str).str.strip().str.title()
    cols = ["name", "obs", "pctwhite", "pctblack"]
    return frame[cols].copy()


def load_ssa_gender() -> pd.DataFrame:
    """Return name, female_share computed from SSA mirror percents.

    `percent` is a name's share of births of its own sex in a year; boy and
    girl yearly totals are close enough that summed percents approximate the
    name's overall female share well enough to gate strongly-gendered names.
    Swap in a raw-count SSA file (name, sex, count) for an exact share.
    """
    raw = RAW_DIR / "ssa_baby_names.csv"
    _download(SSA_GENDER_URL, raw)
    frame = pd.read_csv(raw)
    frame["name"] = frame["name"].astype(str).str.strip().str.title()
    frame["sex"] = frame["sex"].str.lower().map({"girl": "F", "boy": "M"})
    weight = frame.groupby(["name", "sex"])["percent"].sum().unstack(fill_value=0.0)
    weight["female_share"] = weight.get("F", 0.0) / (
        weight.get("F", 0.0) + weight.get("M", 0.0)
    )
    return weight.reset_index()[["name", "female_share"]]


def gender_label(female_share: float, woman_min: float, man_max: float) -> str | None:
    if female_share >= woman_min:
        return "woman"
    if female_share <= man_max:
        return "man"
    return None  # not clearly gendered -> excluded


# Race-specific frequency floors. White and Black sit on opposite ends of the
# frequency-vs-specificity trade-off: ~100%-White names are common, so a high
# floor removes only rare ethnically-marked names (Seamus, Bjorn); the most
# Black-distinctive names (Lakisha, Tyrone) are inherently rare, so the same
# floor would delete them and leave only low-specificity common names. A single
# uniform floor cannot serve both, so each race gets its own.
DEFAULT_OBS_FLOOR = {"White": 1000, "Black": 25}


def build(
    source: NameSource,
    per_cell: int,
    obs_floor: dict[str, int],
    woman_min: float,
    man_max: float,
) -> pd.DataFrame:
    names = load_tzioumis(source)
    gender = load_ssa_gender()
    merged = names.merge(gender, on="name", how="inner")
    merged["gender"] = merged["female_share"].apply(
        lambda s: gender_label(s, woman_min, man_max)
    )
    merged = merged[merged["gender"].notna()]

    rows: list[pd.DataFrame] = []
    for race in RACES:
        score_col = source.race_specificity(race)
        floor = obs_floor[race]
        for gender_lbl in GENDERS:
            cell = merged[
                (merged["gender"] == gender_lbl) & (merged["obs"] >= floor)
            ].copy()
            cell["race"] = race
            cell["subgroup"] = f"{race.lower()}_{gender_lbl}"
            cell["source"] = source.key
            cell["race_specificity"] = cell[score_col]
            cell = cell.sort_values("race_specificity", ascending=False).head(per_cell)
            if len(cell) < per_cell:
                print(
                    f"  WARNING: {race} {gender_lbl}: only {len(cell)} names "
                    f"available (wanted {per_cell})"
                )
            rows.append(cell)

    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "name",
            "race",
            "gender",
            "subgroup",
            "source",
            "race_specificity",
            "female_share",
            "obs",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tzioumis", choices=sorted(SOURCES))
    parser.add_argument("--per-cell", type=int, default=50)
    parser.add_argument(
        "--white-min-obs",
        type=int,
        default=DEFAULT_OBS_FLOOR["White"],
        help="Frequency floor for White cells (high: drops rare ethnic names).",
    )
    parser.add_argument(
        "--black-min-obs",
        type=int,
        default=DEFAULT_OBS_FLOOR["Black"],
        help="Frequency floor for Black cells (low: keeps rare distinctive names).",
    )
    parser.add_argument("--woman-min", type=float, default=0.9)
    parser.add_argument("--man-max", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = SOURCES[args.source]
    out_path = Path(args.out) if args.out else NAMES_DIR / f"names_{source.key}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = build(
        source,
        per_cell=args.per_cell,
        obs_floor={"White": args.white_min_obs, "Black": args.black_min_obs},
        woman_min=args.woman_min,
        man_max=args.man_max,
    )
    table.to_csv(out_path, index=False)

    print(f"\nWrote {len(table)} names to {out_path}")
    counts = table.groupby("subgroup").size().to_dict()
    print("Per-cell counts:", counts)
    for subgroup in sorted(table["subgroup"].unique()):
        sample = table[table["subgroup"] == subgroup]["name"].head(8).tolist()
        print(f"  {subgroup:14s} e.g. {', '.join(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
