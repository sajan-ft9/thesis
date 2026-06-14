"""Configurable image transforms.

Training transforms apply light, label-preserving augmentation (random crop,
horizontal flip, small rotation, brightness/contrast jitter). Validation/test
transforms are strictly deterministic (resize + normalise only) so evaluation is
reproducible and unbiased. Everything is driven by :class:`~src.config.Config`.
"""

from __future__ import annotations

from typing import Any

import torchvision.transforms as T

from .config import Config

__all__ = ["build_train_transform", "build_eval_transform", "build_transforms"]


def build_train_transform(cfg: Config) -> T.Compose:
    """Construct the (augmenting) training transform pipeline from config."""
    t: dict[str, Any] = cfg.transforms.train
    img_size = cfg.data.img_size
    resize = int(t.get("resize", img_size + 32))

    ops: list[Any] = [T.Resize((resize, resize))]
    if t.get("random_crop", True):
        ops.append(T.RandomCrop(img_size))
    else:
        ops.append(T.Resize((img_size, img_size)))
    flip_p = float(t.get("horizontal_flip", 0.5))
    if flip_p > 0:
        ops.append(T.RandomHorizontalFlip(p=flip_p))
    rot = float(t.get("rotation_degrees", 0))
    if rot > 0:
        ops.append(T.RandomRotation(rot))
    brightness = float(t.get("brightness", 0.0))
    contrast = float(t.get("contrast", 0.0))
    if brightness > 0 or contrast > 0:
        ops.append(T.ColorJitter(brightness=brightness, contrast=contrast))
    ops.append(T.ToTensor())
    ops.append(T.Normalize(cfg.transforms.norm_mean, cfg.transforms.norm_std))
    return T.Compose(ops)


def build_eval_transform(cfg: Config) -> T.Compose:
    """Construct the deterministic evaluation/inference transform pipeline."""
    eval_cfg: dict[str, Any] = cfg.transforms.eval
    size = int(eval_cfg.get("resize", cfg.data.img_size))
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(cfg.transforms.norm_mean, cfg.transforms.norm_std),
        ]
    )


def build_transforms(cfg: Config) -> tuple[T.Compose, T.Compose]:
    """Return ``(train_transform, eval_transform)`` for a config."""
    return build_train_transform(cfg), build_eval_transform(cfg)
