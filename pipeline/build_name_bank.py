#!/usr/bin/env python3
"""Build the Arm-B name instance bank with joined demographic covariates.

The two-arm sampling design (see plan: two-arm cue sampling) treats names as
*sampled instances* of demographic groups, fit downstream with a mixed-effects
model: ``stance ~ group + (1 | name nested in group) + covariates``. For that
model to be fittable, every name must carry instance-level covariates. The
Tonneau name lists in ``names.csv`` were transcribed from the paper's appendix
without those covariates, so this script joins them back on from published
sources.

Inputs (all read-only, committed under the repo):
  * ``data/input/names/names.csv`` -- the Tonneau pool (per-source rows); the
    bank is its unique ``(subgroup, name)`` set, with ``n_sources`` = how many of
    the three source lists each name appears in.
  * ``data/reference/external/rosenman_first_nameRaceProbs.tab`` -- Rosenman et
    al. 2023 P(race | first name); the PRIMARY covariate (``cov_p_group``),
    100% coverage of the Tonneau names.
  * ``data/reference/external/tzioumis_firstnames.csv`` -- Tzioumis 2018 name
    frequency + P(race); supplies ``cov_freq`` (population frequency) and a
    cross-check P (``cov_p_group_tz``); partial coverage (~62-74%).
  * ``results/cue_probe/name_scores.csv`` (optional) -- per-name legibility from
    the cue probe; supplies ``cov_probe_recall`` / ``cov_probe_refusal`` so the
    illegible groups (see plan section 5) carry an empirical legibility signal.

Output: ``data/input/names/name_bank.csv``, one row per unique name, deterministic
and offline once the reference tables are cached. Prints per-group covariate
coverage so partial joins are visible, not hidden.

Usage:
    python pipeline/build_name_bank.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from config import INPUT_DIR, REFERENCE_DIR, REPO_ROOT

DEFAULT_POOL_CSV = INPUT_DIR / "names" / "names.csv"
DEFAULT_ROSENMAN_TAB = REFERENCE_DIR / "external" / "rosenman_first_nameRaceProbs.tab"
DEFAULT_TZIOUMIS_CSV = REFERENCE_DIR / "external" / "tzioumis_firstnames.csv"
DEFAULT_PROBE_CSV = REPO_ROOT / "results" / "cue_probe" / "name_scores.csv"
DEFAULT_OUT_CSV = INPUT_DIR / "names" / "name_bank.csv"

# subgroup race token -> (Rosenman column, Tzioumis percent column)
RACE_COLS = {
    "black": ("bla", "pctblack"),
    "white": ("whi", "pctwhite"),
}

FIELDNAMES = [
    "subgroup",
    "race",
    "gender",
    "name",
    "n_sources",
    "name_length",
    "cov_p_group",        # Rosenman P(perceived race | name) -- primary, 100% cov
    "cov_p_group_tz",     # Tzioumis P(race | name) -- cross-check, partial
    "cov_freq",           # Tzioumis population frequency (obs) -- partial
    "cov_probe_recall",   # cue-probe race_recall -- legibility, if probed
    "cov_probe_refusal",  # cue-probe race_abstain -- legibility, if probed
]


def _norm(name: str) -> str:
    return name.strip().strip('"').upper()


def load_unique_names(pool_csv: Path) -> list[dict]:
    """Unique ``(subgroup, name)`` rows with ``n_sources`` and race/gender."""

    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    meta: dict[tuple[str, str], dict] = {}
    with pool_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sg = row["subgroup"].strip()
            name = row["name"].strip()
            key = (sg, name.lower())
            sources[key].add(row["source"].strip())
            meta.setdefault(
                key,
                {
                    "subgroup": sg,
                    "race": row["race"].strip(),
                    "gender": row["gender"].strip(),
                    "name": name,
                },
            )
    rows: list[dict] = []
    for key, info in meta.items():
        info = dict(info)
        info["n_sources"] = len(sources[key])
        rows.append(info)
    # Stable order: subgroup, then name.
    rows.sort(key=lambda r: (r["subgroup"], r["name"].lower()))
    return rows


def load_rosenman(path: Path) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            table[_norm(row["name"])] = {
                k: float(row[k]) for k in ("whi", "bla", "his", "asi", "oth")
            }
    return table


def load_tzioumis(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {_norm(row["firstname"]): row for row in csv.DictReader(handle)}


def load_probe(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Per ``(subgroup, name)`` legibility, averaged over source lists."""

    if not path.exists():
        return {}
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            agg[(row["subgroup"].strip(), row["name"].strip().lower())].append(row)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in agg.items():
        recalls = [float(r["race_recall"]) for r in rows if r.get("race_recall")]
        abstains = [float(r["race_abstain"]) for r in rows if r.get("race_abstain")]
        out[key] = {
            "recall": sum(recalls) / len(recalls) if recalls else None,
            "refusal": sum(abstains) / len(abstains) if abstains else None,
        }
    return out


def build_rows(
    names: list[dict],
    rosenman: dict[str, dict[str, float]],
    tzioumis: dict[str, dict[str, str]],
    probe: dict[tuple[str, str], dict[str, float]],
) -> list[dict]:
    rows: list[dict] = []
    for info in names:
        norm = _norm(info["name"])
        ros_col, tz_col = RACE_COLS[info["race"]]

        p_group = ""
        if norm in rosenman:
            p_group = f"{rosenman[norm][ros_col]:.6f}"

        p_group_tz = ""
        freq = ""
        if norm in tzioumis:
            tz = tzioumis[norm]
            # Tzioumis percents are 0-100; store as a 0-1 probability.
            if tz.get(tz_col):
                p_group_tz = f"{float(tz[tz_col]) / 100.0:.6f}"
            if tz.get("obs"):
                freq = tz["obs"]

        leg = probe.get((info["subgroup"], info["name"].lower()), {})
        recall = leg.get("recall")
        refusal = leg.get("refusal")

        rows.append(
            {
                "subgroup": info["subgroup"],
                "race": info["race"],
                "gender": info["gender"],
                "name": info["name"],
                "n_sources": info["n_sources"],
                "name_length": len(info["name"]),
                "cov_p_group": p_group,
                "cov_p_group_tz": p_group_tz,
                "cov_freq": freq,
                "cov_probe_recall": "" if recall is None else f"{recall:.4f}",
                "cov_probe_refusal": "" if refusal is None else f"{refusal:.4f}",
            }
        )
    return rows


def coverage_report(rows: list[dict]) -> str:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["subgroup"]].append(row)
    lines = ["Per-group covariate coverage (non-blank):"]
    for sg in sorted(by_group):
        grp = by_group[sg]
        n = len(grp)

        def pct(col: str) -> str:
            hit = sum(1 for r in grp if r[col] != "")
            return f"{hit}/{n} ({100 * hit // n}%)"

        lines.append(
            f"  {sg}: p_group={pct('cov_p_group')} "
            f"freq={pct('cov_freq')} probe={pct('cov_probe_recall')}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=str(DEFAULT_POOL_CSV))
    parser.add_argument("--rosenman", default=str(DEFAULT_ROSENMAN_TAB))
    parser.add_argument("--tzioumis", default=str(DEFAULT_TZIOUMIS_CSV))
    parser.add_argument("--probe", default=str(DEFAULT_PROBE_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = load_unique_names(Path(args.pool))
    rosenman = load_rosenman(Path(args.rosenman))
    tzioumis = load_tzioumis(Path(args.tzioumis))
    probe = load_probe(Path(args.probe))

    rows = build_rows(names, rosenman, tzioumis, probe)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["subgroup"]] += 1
    print(f"Wrote {len(rows)} name-bank rows to {out_path}")
    print(f"Per subgroup: {dict(sorted(counts.items()))}")
    print(coverage_report(rows))
    if not probe:
        print("(cue-probe legibility not found -- cov_probe_* left blank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
