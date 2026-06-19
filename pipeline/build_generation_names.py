#!/usr/bin/env python3
"""Sample the main-run name cues from the cue-legibility probe pool.

The probe (``05_run_cue_probe.py``) reads the full Tonneau et al. name set from
``data/input/names/names.csv`` (150 names per subgroup). The main generation run
and the PCT arm read a much smaller subset from
``data/input/names/names_generation.csv`` so the run stays tractable. This script
draws that subset *from the probed pool* with a fixed seed, so every name the
main run uses is one the probe has a legibility readout for.

Sampling is uniform over the unique ``(subgroup, name)`` pairs in ``names.csv``
(a name appearing in more than one source list is not over-weighted), is seeded
and deterministic, and is done independently per subgroup. The output preserves
the ``names.csv`` schema (``source, race, gender, subgroup, name``) so the
downstream loaders (``name_cues_from_csv``) are unchanged. Within each subgroup
rows keep their original ``names.csv`` order so the file is stable.

Usage:
    python pipeline/build_generation_names.py                 # 10/group, seed 0
    python pipeline/build_generation_names.py --per-group 5 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from config import INPUT_DIR

DEFAULT_POOL_CSV = INPUT_DIR / "names" / "names.csv"
DEFAULT_OUT_CSV = INPUT_DIR / "names" / "names_generation.csv"
FIELDNAMES = ["source", "race", "gender", "subgroup", "name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=str(DEFAULT_POOL_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_rows(pool_rows: list[dict], per_group: int, seed: int) -> list[dict]:
    """Seeded draw of ``per_group`` unique names per subgroup, in pool order."""

    by_group: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for row in pool_rows:
        subgroup = row["subgroup"].strip()
        name = row["name"].strip()
        key = (subgroup, name.lower())
        if key in seen:
            continue
        seen.add(key)
        by_group.setdefault(subgroup, []).append(row)

    selected: list[dict] = []
    for subgroup in sorted(by_group):
        pool = by_group[subgroup]
        take = min(per_group, len(pool))
        rng = random.Random(f"{seed}:{subgroup}")
        drawn = rng.sample(pool, take)
        # Restore pool order so the written file does not depend on draw order.
        drawn.sort(key=pool.index)
        selected.extend(drawn)
    return selected


def main() -> int:
    args = parse_args()
    with Path(args.pool).open(newline="", encoding="utf-8-sig") as handle:
        pool_rows = list(csv.DictReader(handle))

    rows = sample_rows(pool_rows, args.per_group, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDNAMES})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["subgroup"]] = counts.get(row["subgroup"], 0) + 1
    print(f"Wrote {len(rows)} sampled names to {out_path}")
    print(f"Per subgroup: {dict(sorted(counts.items()))} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
