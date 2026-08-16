"""Fresh-process worker for reproducible process-RSS measurements.

This module is invoked by ``src.memory_profile``.  Keeping one model variant per
process prevents allocator caches and imported backend state from contaminating
comparisons between FP32 and quantized inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psutil
import torch

from .benchmarking import benchmark_peak_rss, file_size_mb
from .dataset import build_dataloaders
from .inference import collect_predictions, load_checkpoint
from .quantize import dynamic_quantize, pick_backend, static_quantize
from .utils import save_json


def _save_quantized(model: torch.nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.jit.save(torch.jit.script(model), str(path))
    except Exception:  # noqa: BLE001 - state-dict artifact remains valid fallback
        path = path.with_suffix(".pth")
        torch.save(model.state_dict(), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("fp32", "dynamic", "static"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    cfg.data.num_workers = 0
    data = build_dataloaders(cfg, seed=cfg.seed, drop_last=False)
    artifact = Path(args.artifact)
    quant_meta: dict = {}

    if args.variant == "fp32":
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if Path(args.checkpoint).resolve() != artifact.resolve():
            torch.save(model.state_dict(), artifact)
    elif args.variant == "dynamic":
        model = dynamic_quantize(model, cfg.quantize.backend)
        torch.save(model.state_dict(), artifact)
    else:
        backend = pick_backend(cfg.quantize.backend)
        model = static_quantize(model, data.val_loader, backend, cfg.data.img_size, cfg.quantize.calibration_batches)
        artifact = _save_quantized(model, artifact)
        quant_meta = getattr(model, "_publication_quantization", {})

    baseline = psutil.Process().memory_info().rss / 1e6

    def inference() -> None:
        collect_predictions(model, data.test_loader, device, cfg.evaluate.threshold)

    _, delta = benchmark_peak_rss(inference)
    result = {
        "variant": {"fp32": "FP32", "dynamic": "INT8 dynamic", "static": "INT8 static (PTQ)"}[args.variant],
        "artifact_path": str(artifact),
        "model_size_mb": round(file_size_mb(artifact), 2),
        "artifact_size_mb": round(file_size_mb(artifact), 3),
        "inference_peak_rss_mb": round(delta, 1),
        "rss_baseline_mb": round(baseline, 1),
        "measurement": "fresh subprocess; process RSS delta during full test inference; num_workers=0",
        "quantization": quant_meta,
    }
    save_json(result, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
