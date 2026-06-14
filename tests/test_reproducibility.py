"""Tests that seeding makes data loading and model init reproducible."""

from __future__ import annotations

import torch

from src.dataset import build_dataloaders
from src.models import build_model
from src.utils import seed_everything


def test_first_batch_is_reproducible(cfg) -> None:
    seed_everything(cfg.seed)
    batch_a = next(iter(build_dataloaders(cfg).train_loader))[0]
    seed_everything(cfg.seed)
    batch_b = next(iter(build_dataloaders(cfg).train_loader))[0]
    assert torch.allclose(batch_a, batch_b)


def test_model_init_is_reproducible() -> None:
    seed_everything(7)
    m1 = build_model(name="efficientnet_b0", pretrained=False, freeze_stages=0)
    seed_everything(7)
    m2 = build_model(name="efficientnet_b0", pretrained=False, freeze_stages=0)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
