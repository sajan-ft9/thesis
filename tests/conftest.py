"""Shared pytest fixtures: a tiny synthetic dataset and a CPU test config.

The synthetic images are random with a faint label-dependent bias so models can
fit them quickly. They are NOT chest X-rays; nothing produced from them is a
research result. Fixtures keep the unit tests fast and offline (no ImageNet
weights are downloaded — ``model.pretrained=false`` everywhere).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make the `src` package importable when running `pytest` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402


def _make_split(root: Path, split: str, per_class: int, rng: np.random.Generator) -> None:
    for class_name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8).astype(np.float32)
            arr += 25.0 if label == 1 else -25.0
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
            img.save(class_dir / f"{class_name.lower()}_{i:03d}.png")


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny synthetic dataset once per test session."""
    root = tmp_path_factory.mktemp("synthetic_cxr")
    rng = np.random.default_rng(42)
    _make_split(root, "train", per_class=16, rng=rng)
    _make_split(root, "test", per_class=8, rng=rng)
    return root


@pytest.fixture
def cfg(synthetic_root: Path) -> Config:
    """A small, fast, CPU-only config pointed at the synthetic dataset."""
    return load_config(
        "configs/efficientnet_b0.yaml",
        overrides=[
            f"data.root={synthetic_root}",
            "data.num_workers=0",
            "data.batch_size=4",
            "data.img_size=64",
            "transforms.train.resize=72",
            "transforms.eval.resize=64",
            "model.pretrained=false",
            "device=cpu",
            "train.epochs=1",
            "train.warmup_epochs=1",
            "train.unfreeze_epoch=1",
            "evaluate.bootstrap_n=100",
            "benchmark.warmup=1",
            "benchmark.repeats=3",
            "quantize.calibration_batches=2",
            "explain.num_per_category=1",
        ],
    )
