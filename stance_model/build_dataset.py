#!/usr/bin/env python3
"""Build the supervised stance-regression dataset from the ai-distortion study.

Joins reader-aggregated stance ratings (`writer_stance`, 0-100) onto the actual
paragraph text and the proposition each paragraph is about, producing one row per
(writer, proposition, paragraph_type) example for a cross-encoder that predicts
"how much does this text support its proposition".

Source: paul-rottger/ai-distortion (Roettger et al. 2026), main study.
  - main_phase_1/paragraphs.csv      -> text     (writer / model / edited variants)
  - main_phase_1/propositions.csv    -> proposition string + political leaning
  - main_phase_2/annotations_aggregated.csv -> writer_stance label (reader mean)

writer_stance is DIRECTIONAL: 0 = opposes the proposition, 50 = neutral,
100 = supports. (writer_stance_polarity = |stance-50|*2 is extremity only and is
NOT the training target.)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_BASE = "https://raw.githubusercontent.com/paul-rottger/ai-distortion/main/data"
RAW_DIR = REPO_ROOT / "data" / "reference" / "ai_distortion"
OUT_DIR = REPO_ROOT / "data" / "processed" / "stance_model"

FILES = {
    "paragraphs": "main_phase_1/paragraphs.csv",
    "propositions": "main_phase_1/propositions.csv",
    "annotations": "main_phase_2/annotations_aggregated.csv",
}

# Keys shared by paragraphs.csv and annotations_aggregated.csv. model_name and
# model_input_condition are blank for writer paragraphs, so we fill NaN -> "" on
# both sides before merging to keep those rows.
JOIN_KEYS = [
    "writer_id",
    "proposition_id",
    "paragraph_type",
    "model_name",
    "model_input_condition",
]


def download(force: bool = False) -> dict[str, Path]:
    paths = {}
    for name, rel in FILES.items():
        dest = RAW_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force or not dest.exists():
            url = f"{RAW_BASE}/{rel}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest)
        else:
            print(f"Cached {dest}")
        paths[name] = dest
    return paths


def build(paths: dict[str, Path]) -> pd.DataFrame:
    paragraphs = pd.read_csv(paths["paragraphs"])
    propositions = pd.read_csv(paths["propositions"])
    annotations = pd.read_csv(paths["annotations"])

    for frame in (paragraphs, annotations):
        for key in ("model_name", "model_input_condition"):
            frame[key] = frame[key].fillna("")

    # text + proposition come from paragraphs.csv; the label from annotations.
    label_cols = ["writer_stance", "writer_stance_polarity"]
    ann = annotations[JOIN_KEYS + label_cols].copy()

    merged = paragraphs.merge(ann, on=JOIN_KEYS, how="inner", validate="one_to_one")

    # attach proposition leaning (left/right/neither) for downstream slicing.
    lean = propositions[["proposition_id", "proposition_leaning"]]
    merged = merged.merge(lean, on="proposition_id", how="left")

    merged = merged.rename(columns={"paragraph": "text"})
    merged = merged.dropna(subset=["text", "writer_stance"])
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged[merged["text"].str.len() > 0]

    merged["example_id"] = (
        merged["writer_id"].astype(str)
        + "_"
        + merged["proposition_id"].astype(str)
        + "_"
        + merged["paragraph_type"].astype(str)
    )

    cols = [
        "example_id",
        "writer_id",
        "proposition_id",
        "proposition",
        "proposition_leaning",
        "paragraph_type",
        "model_name",
        "model_input_condition",
        "text",
        "writer_stance",
        "writer_stance_polarity",
    ]
    return merged[cols].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--out", default=str(OUT_DIR / "dataset.csv"))
    args = parser.parse_args()

    paths = download(force=args.force_download)
    df = build(paths)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nWrote {len(df)} examples to {out}")
    print(f"  unique propositions: {df['proposition_id'].nunique()}")
    print(f"  unique writers:      {df['writer_id'].nunique()}")
    print("\nby paragraph_type:")
    print(df["paragraph_type"].value_counts().to_string())
    print("\nwriter_stance summary:")
    print(df["writer_stance"].describe().round(2).to_string())
    print("\nwriter_stance by paragraph_type (mean):")
    print(df.groupby("paragraph_type")["writer_stance"].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
