"""Tests for reproducibility, device and metadata utilities."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils import (
    build_metadata,
    count_parameters,
    get_device,
    load_json,
    save_json,
    seed_everything,
)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    a = (random.random(), float(np.random.rand()), torch.rand(1).item())
    seed_everything(123)
    b = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert a == b


def test_get_device_cpu() -> None:
    assert get_device("cpu").type == "cpu"


def test_count_parameters() -> None:
    model = torch.nn.Linear(10, 2)
    total, trainable = count_parameters(model)
    assert total == trainable == 10 * 2 + 2
    for p in model.parameters():
        p.requires_grad = False
    _, trainable = count_parameters(model)
    assert trainable == 0


def test_build_metadata_has_required_fields() -> None:
    meta = build_metadata(experiment_name="x", config={"a": 1}, seed=7)
    for key in ("experiment_name", "timestamp_utc", "seed", "library_versions", "config"):
        assert key in meta
    assert meta["seed"] == 7


def test_json_roundtrip(tmp_path) -> None:
    obj = {"a": np.int64(3), "b": np.float32(1.5), "c": [1, 2, 3]}
    path = save_json(obj, tmp_path / "x.json")
    loaded = load_json(path)
    assert loaded["a"] == 3 and loaded["b"] == 1.5 and loaded["c"] == [1, 2, 3]
