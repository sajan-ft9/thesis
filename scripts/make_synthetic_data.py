"""Generate a TINY SYNTHETIC dataset to smoke-test the pipeline end-to-end.

WARNING: these are random images with a weak, artificial class signal. They exist
ONLY to verify that the code runs without errors and produces all artifacts. They
are NOT chest X-rays and any numbers produced from them are meaningless and must
never be reported as research results.

Usage:
    python scripts/make_synthetic_data.py --out data/synthetic --per-class-train 40 --per-class-test 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _make_image(rng: np.random.Generator, label: int, size: int = 256) -> Image.Image:
    """Random RGB image with a faint label-dependent bias so a model can fit it."""
    base = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8).astype(np.float32)
    # Inject a weak, label-dependent intensity gradient (artificial signal).
    bias = 30.0 if label == 1 else -30.0
    yy = np.linspace(-1, 1, size)[:, None]
    base += bias * yy
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def _write_split(out_dir: Path, split: str, per_class: int, rng: np.random.Generator) -> None:
    for class_name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        class_dir = out_dir / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            _make_image(rng, label).save(class_dir / f"{class_name.lower()}_{i:04d}.jpeg", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--per-class-train", type=int, default=40)
    parser.add_argument("--per-class-test", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    _write_split(out_dir, "train", args.per_class_train, rng)
    _write_split(out_dir, "test", args.per_class_test, rng)
    print(f"[synthetic] wrote dataset to {out_dir} (NON-RESEARCH; for smoke testing only)")


if __name__ == "__main__":
    main()
