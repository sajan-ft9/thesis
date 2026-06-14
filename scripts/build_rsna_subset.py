"""Build a balanced Normal-vs-Pneumonia subset from the RSNA mirror for EXTERNAL,
inference-only validation of the Kermany-trained model.

The RSNA mirror stores DICOM (.dcm) images, so this script performs a one-time
DICOM->PNG conversion (kept isolated here; the core pipeline still reads PNG and is
unchanged). Mapping (deliberately simple, exploratory — not a clinical mapping):
    RSNA "Normal"        -> NORMAL
    RSNA "Lung Opacity"  -> PNEUMONIA
    RSNA "No Lung Opacity / Not Normal" -> EXCLUDED

Selects all Lung Opacity patients and an equal, seeded random sample of Normal patients,
converts each to a 224x224 RGB PNG, and writes:
    data/processed/rsna_external/{NORMAL,PNEUMONIA}/<patientId>.png

Usage:
    python scripts/build_rsna_subset.py --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

RAW = Path("data/raw/rsna")
OUT = Path("data/processed/rsna_external")
IMG_DIRS = ["stage_2_train_images_0", "stage_2_train_images_1", "stage_2_train_images_2"]


def _index_images() -> dict[str, Path]:
    """Map patientId -> DICOM path across the RSNA train image directories."""
    index: dict[str, Path] = {}
    for d in IMG_DIRS:
        base = RAW / d
        if not base.is_dir():
            continue
        for dcm in base.glob("*.dcm"):
            index[dcm.stem] = dcm
    return index


def _dcm_to_png(path: Path, out_path: Path, size: int = 224) -> None:
    """Convert a DICOM chest X-ray to a size x size RGB PNG (8-bit, MONOCHROME-aware)."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype("float32")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = 255.0 - arr  # invert so bones are bright, matching standard display
    img = Image.fromarray(arr.astype("uint8")).convert("RGB").resize((size, size))
    img.save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class-cap", type=int, default=0,
                        help="Optional cap per class (0 = use all available, balanced).")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    dci = pd.read_csv(RAW / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    images = _index_images()
    print(f"[rsna] indexed {len(images)} DICOM images; {len(dci)} labelled patients")

    normal = [p for p in dci[dci["class"] == "Normal"]["patientId"] if p in images]
    pneumonia = [p for p in dci[dci["class"] == "Lung Opacity"]["patientId"] if p in images]
    print(f"[rsna] available -> Normal: {len(normal)} | Pneumonia (Lung Opacity): {len(pneumonia)}")
    if not normal or not pneumonia:
        raise SystemExit("[rsna] ABORT: images not found — download likely incomplete.")

    n = min(len(normal), len(pneumonia))
    if args.per_class_cap > 0:
        n = min(n, args.per_class_cap)
    rng = np.random.default_rng(args.seed)
    normal_sel = sorted(rng.choice(normal, size=n, replace=False).tolist())
    pneumonia_sel = sorted(rng.choice(pneumonia, size=n, replace=False).tolist())

    for cls, ids in (("NORMAL", normal_sel), ("PNEUMONIA", pneumonia_sel)):
        d = OUT / cls
        d.mkdir(parents=True, exist_ok=True)
        for i, pid in enumerate(ids, 1):
            _dcm_to_png(images[pid], d / f"{pid}.png", size=args.size)
            if i % 1000 == 0:
                print(f"[rsna]   {cls}: converted {i}/{len(ids)}")

    print(f"[rsna] balanced subset written to {OUT} -> {n} Normal + {n} Pneumonia "
          f"({2 * n} images, {args.size}px, seed={args.seed})")


if __name__ == "__main__":
    main()
