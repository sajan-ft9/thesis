"""Download the Kermany Chest X-Ray Pneumonia dataset and export it to folders.

Source: the Hugging Face mirror ``hf-vision/chest-xray-pneumonia`` (CC BY 4.0),
which stores the exact Kermany splits (train 5216, val 16, test 624) as parquet
with embedded images. This avoids needing Kaggle credentials.

The images are written into the layout the pipeline expects:

    data/raw/chest_xray/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg

The tiny official validation split (16 images) is intentionally not exported; the
pipeline carves a stratified validation set from train (see configs/base.yaml).

Usage:
    python scripts/fetch_kermany_hf.py --out data/raw/chest_xray
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

REPO = "hf-vision/chest-xray-pneumonia"
LABELS = {0: "NORMAL", 1: "PNEUMONIA"}


def export_split(split, out_root: Path, split_name: str) -> dict[str, int]:
    counts = {"NORMAL": 0, "PNEUMONIA": 0}
    for i, example in enumerate(split):
        label = LABELS[int(example["label"])]
        out_dir = out_root / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        example["image"].convert("RGB").save(
            out_dir / f"{split_name}_{label.lower()}_{i:05d}.jpeg", quality=95
        )
        counts[label] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/chest_xray")
    args = parser.parse_args()

    print(f"[fetch] loading {REPO} (≈1.2 GB download on first run)...")
    ds = load_dataset(REPO)
    out_root = Path(args.out)

    for src, dst in (("train", "train"), ("test", "test")):
        counts = export_split(ds[src], out_root, dst)
        print(f"[fetch] {dst}: {counts}")
    print(f"[fetch] done -> {out_root}")


if __name__ == "__main__":
    main()
