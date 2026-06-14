"""Tests for dataset scanning, splitting, integrity checks and loaders."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from src.dataset import (
    build_dataloaders,
    class_distribution,
    find_duplicates,
    scan_split,
    stratified_split,
    validate_dataset,
    verify_no_leakage,
)


def test_scan_split_finds_labeled_images(synthetic_root: Path) -> None:
    samples = scan_split(synthetic_root / "train")
    assert len(samples) == 32  # 16 per class
    labels = {label for _, label in samples}
    assert labels == {0, 1}


def test_class_distribution(synthetic_root: Path) -> None:
    dist = class_distribution(scan_split(synthetic_root / "train"))
    assert dist == {"Normal": 16, "Pneumonia": 16}


def test_stratified_split_is_disjoint_and_balanced(synthetic_root: Path) -> None:
    samples = scan_split(synthetic_root / "train")
    train, val = stratified_split(samples, val_split=0.25, seed=42)
    assert len(val) == 8 and len(train) == 24
    assert set(p for p, _ in train).isdisjoint(p for p, _ in val)
    assert class_distribution(val) == {"Normal": 4, "Pneumonia": 4}


def test_find_duplicates_detects_copy(synthetic_root: Path, tmp_path: Path) -> None:
    samples = scan_split(synthetic_root / "train")
    original = samples[0][0]
    dup = tmp_path / "dup.png"
    shutil.copy(original, dup)
    dups = find_duplicates(samples + [(str(dup), samples[0][1])])
    assert any(str(dup) in paths for paths in dups.values())


def test_verify_no_leakage(synthetic_root: Path) -> None:
    samples = scan_split(synthetic_root / "train")
    train, val = stratified_split(samples, 0.25, 42)
    clean = verify_no_leakage({"train": train, "val": val})
    assert clean["clean"] is True
    leaked = verify_no_leakage({"train": train, "val": train[:2] + val})
    assert leaked["clean"] is False
    assert leaked["path_leaks"]


def test_build_dataloaders_shapes(cfg) -> None:
    bundle = build_dataloaders(cfg, drop_last=False)
    imgs, labels = next(iter(bundle.train_loader))
    assert imgs.shape[1:] == (3, cfg.data.img_size, cfg.data.img_size)
    assert imgs.dtype == torch.float32
    assert labels.dtype == torch.float32


def test_validate_dataset_report(cfg) -> None:
    report = validate_dataset(cfg)
    assert report["leakage"]["clean"] is True
    assert set(report["split_sizes"]) == {"train", "val", "test"}
