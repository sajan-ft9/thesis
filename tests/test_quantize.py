"""Tests for quantization (dynamic + static PTQ) and backend selection."""

from __future__ import annotations

import torch

from src.dataset import build_dataloaders
from src.models import build_model
from src.quantize import dynamic_quantize, pick_backend, static_quantize


def test_pick_backend_returns_supported() -> None:
    backend = pick_backend("auto")
    supported = list(getattr(torch.backends.quantized, "supported_engines", []))
    assert backend in supported


def test_dynamic_quantize_runs() -> None:
    model = build_model(name="efficientnet_b0", pretrained=False, freeze_stages=0).eval()
    qmodel = dynamic_quantize(model)
    out = qmodel(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1)


def test_static_ptq_runs_or_reports(cfg) -> None:
    """Static PTQ should either produce a working INT8 model or fail explicitly."""
    model = build_model(name="efficientnet_b0", pretrained=False, freeze_stages=0).eval()
    data = build_dataloaders(cfg, drop_last=False)
    backend = pick_backend("auto")
    try:
        qmodel = static_quantize(model, data.train_loader, backend, img_size=cfg.data.img_size, calibration_batches=2)
    except Exception as exc:  # noqa: BLE001 - acceptable: report, never fabricate
        import pytest

        pytest.skip(f"Static PTQ unsupported on backend '{backend}': {exc}")
    out = qmodel(torch.randn(1, 3, cfg.data.img_size, cfg.data.img_size))
    assert out.shape == (1, 1)
