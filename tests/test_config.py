"""Tests for hierarchical config loading and overrides."""

from __future__ import annotations

import pytest

from src.config import Config, load_config


def test_defaults_load() -> None:
    cfg = load_config("configs/base.yaml")
    assert cfg.model.name == "efficientnet_b0"
    assert cfg.data.img_size == 224
    assert cfg.seed == 42


def test_extends_merges_parent() -> None:
    cfg = load_config("configs/resnet18.yaml")
    assert cfg.model.name == "resnet18"          # from child
    assert cfg.train.epochs == 20                # inherited from base
    assert cfg.experiment_name == "resnet18"


def test_override_types_preserved() -> None:
    cfg = load_config(
        "configs/efficientnet_b0.yaml",
        overrides=["train.epochs=3", "model.pretrained=false", "train.lr=0.001"],
    )
    assert cfg.train.epochs == 3 and isinstance(cfg.train.epochs, int)
    assert cfg.model.pretrained is False
    assert cfg.train.lr == pytest.approx(0.001)


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"not_a_real_key": 1}})


def test_roundtrip_to_dict() -> None:
    cfg = load_config("configs/efficientnet_b0.yaml")
    restored = Config.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()
