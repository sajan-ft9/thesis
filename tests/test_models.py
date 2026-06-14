"""Tests for the model factory across all supported architectures."""

from __future__ import annotations

import pytest
import torch

from src.models import SUPPORTED_MODELS, build_model
from src.utils import count_parameters


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_forward_output_shape(name: str) -> None:
    model = build_model(name=name, pretrained=False, freeze_stages=0).eval()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1)


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_freeze_then_unfreeze(name: str) -> None:
    model = build_model(name=name, pretrained=False, freeze_stages=3)
    _, trainable_frozen = count_parameters(model)
    total, _ = count_parameters(model)
    assert trainable_frozen < total  # something is frozen
    model.unfreeze_all()
    _, trainable_all = count_parameters(model)
    assert trainable_all == total


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_cam_target_layers_present(name: str) -> None:
    model = build_model(name=name, pretrained=False, freeze_stages=0)
    layers = model.get_cam_target_layers()
    assert isinstance(layers, list) and len(layers) >= 1


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError):
        build_model(name="not_a_model", pretrained=False)
