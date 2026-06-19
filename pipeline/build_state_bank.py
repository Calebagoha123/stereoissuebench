#!/usr/bin/env python3
"""Build the Arm-B state instance bank (red / swing / blue, full membership).

In the two-arm design, states are *sampled instances* of partisan categories,
the same way names are instances of demographic groups. The old code hand-picked
3 states per category; this bank uses the full membership of all 50 states so the
category is represented broadly and no single state is welded to a task.

Classification rule (2024 cycle):
  * ``swing`` = the seven standard 2024 battlegrounds: AZ, GA, MI, NV, NC, PA, WI.
  * every other state is ``blue`` if its 2024 presidential two-party margin was
    Democratic (> 0) and ``red`` if Republican (< 0).

``cov_margin_2024`` is the 2024 presidential margin in points, Democratic minus
Republican (positive = Democratic lean). It is an instance-level CONTROL
covariate, not the grouping variable, so approximate values do not affect the
red/swing/blue assignment -- but they should be reconciled against a citable
certified-results source before any state-margin analysis is reported. The
category assignment itself is robust.

Output: ``data/input/states/state_bank.csv``.

Usage:
    python pipeline/build_state_bank.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import INPUT_DIR

DEFAULT_OUT_CSV = INPUT_DIR / "states" / "state_bank.csv"
FIELDNAMES = ["state", "category", "cov_margin_2024"]

SWING = {"Arizona", "Georgia", "Michigan", "Nevada", "North Carolina", "Pennsylvania", "Wisconsin"}

# 2024 presidential two-party margin, Democratic minus Republican (points).
# Best-effort certified-ish values; verify against an official source before
# reporting any margin-level analysis. Sign (hence red/blue) is reliable.
MARGIN_2024: dict[str, float] = {
    "Alabama": -30.6, "Alaska": -13.1, "Arizona": -5.5, "Arkansas": -31.0,
    "California": 20.2, "Colorado": 11.0, "Connecticut": 14.5, "Delaware": 14.9,
    "Florida": -13.1, "Georgia": -2.2, "Hawaii": 23.4, "Idaho": -37.0,
    "Illinois": 11.0, "Indiana": -19.0, "Iowa": -13.2, "Kansas": -16.1,
    "Kentucky": -30.5, "Louisiana": -22.1, "Maine": 7.0, "Maryland": 26.0,
    "Massachusetts": 25.5, "Michigan": -1.4, "Minnesota": 4.3, "Mississippi": -23.4,
    "Missouri": -18.5, "Montana": -20.1, "Nebraska": -20.5, "Nevada": -3.1,
    "New Hampshire": 2.8, "New Jersey": 5.9, "New Mexico": 6.1, "New York": 12.6,
    "North Carolina": -3.3, "North Dakota": -36.4, "Ohio": -11.2, "Oklahoma": -33.5,
    "Oregon": 14.9, "Pennsylvania": -1.7, "Rhode Island": 14.0, "South Carolina": -18.0,
    "South Dakota": -29.0, "Tennessee": -30.1, "Texas": -13.7, "Utah": -21.5,
    "Vermont": 32.4, "Virginia": 5.8, "Washington": 18.8, "West Virginia": -41.9,
    "Wisconsin": -0.9, "Wyoming": -45.7,
}


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for state in sorted(MARGIN_2024):
        margin = MARGIN_2024[state]
        if state in SWING:
            category = "swing_state"
        elif margin > 0:
            category = "blue_state"
        else:
            category = "red_state"
        rows.append(
            {"state": state, "category": category, "cov_margin_2024": f"{margin:.1f}"}
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV))
    args = parser.parse_args()

    rows = build_rows()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"Wrote {len(rows)} states to {out_path}")
    print(f"Per category: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
