"""Tests for configurable transforms."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.transforms import build_eval_transform, build_transforms


def _dummy_image() -> Image.Image:
    arr = (np.random.default_rng(0).random((80, 80, 3)) * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def test_transforms_output_shape(cfg) -> None:
    train_t, eval_t = build_transforms(cfg)
    img = _dummy_image()
    out_train = train_t(img)
    out_eval = eval_t(img)
    assert out_train.shape == (3, cfg.data.img_size, cfg.data.img_size)
    assert out_eval.shape == (3, cfg.data.img_size, cfg.data.img_size)
    assert isinstance(out_train, torch.Tensor)


def test_eval_transform_is_deterministic(cfg) -> None:
    eval_t = build_eval_transform(cfg)
    img = _dummy_image()
    assert torch.allclose(eval_t(img), eval_t(img))
