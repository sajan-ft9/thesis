"""Single-measurement memory worker, run as a fresh subprocess.

Two things make a bare, single-sample memory measurement untrustworthy here:

1. **Cross-contamination.** Measuring several variants back-to-back in one
   long-lived process lets an earlier variant's allocator pages get reused by
   a later one, biasing the comparison. Every measurement in this module runs
   in its own fresh subprocess instead.
2. **Sampling misses.** The polling-thread peak-RSS measurement used
   elsewhere in this codebase (:func:`src.benchmarking.benchmark_peak_rss`)
   can miss a transient spike that happens between polls -- on this machine
   that produced swings of several hundred MB for the *identical* variant
   run twice. This module instead reads the kernel's own high-water-mark
   counter (:func:`src.benchmarking.measure_rss_delta`), which cannot miss a
   spike, and reports several repeats so the remaining run-to-run variance
   (real OS/allocator noise, not a measurement artifact) is visible as a
   spread rather than hidden behind one number.

``src.memory_profile`` calls this module several times per variant/scenario
and aggregates with the median; see ``aggregate_median`` there.

Run:
    python -m src.memory_worker runtime --checkpoint models/x_best.pth --variant "FP32"
    python -m src.memory_worker dataset --config configs/efficientnet_b0.yaml --scenario stream
Prints one JSON line to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .benchmarking import file_size_mb, measure_rss_delta
from .config import add_config_cli_args, load_config
from .dataset import ChestXRayDataset, build_dataloaders, scan_split
from .inference import collect_predictions, load_checkpoint
from .quantize import dynamic_quantize, pick_backend, static_quantize
from .transforms import build_eval_transform
from .utils import ensure_dir, get_logger

logger = get_logger("memworker")

VARIANTS = ("FP32", "INT8 dynamic", "INT8 static (PTQ)")
SCENARIOS = ("stream", "naive")

__all__ = ["VARIANTS", "SCENARIOS", "measure_runtime_variant", "measure_dataset_scenario"]


def _build_and_save(model_fp32, label: str, cfg, data, artifact_path: Path) -> torch.nn.Module:
    if label == "FP32":
        model = model_fp32
        torch.save(model.state_dict(), artifact_path)
    elif label == "INT8 dynamic":
        model = dynamic_quantize(model_fp32, cfg.quantize.backend)
        torch.save(model.state_dict(), artifact_path)
    elif label == "INT8 static (PTQ)":
        backend = pick_backend(cfg.quantize.backend)
        model = static_quantize(model_fp32, data.val_loader, backend, cfg.data.img_size, cfg.quantize.calibration_batches)
        try:
            torch.jit.save(torch.jit.script(model), str(artifact_path))
        except Exception:  # noqa: BLE001 - scripting is best-effort
            torch.save(model.state_dict(), artifact_path)
    else:
        raise ValueError(f"unknown variant: {label!r} (expected one of {VARIANTS})")
    return model


def measure_runtime_variant(checkpoint_path: str | Path, label: str) -> dict[str, float | str]:
    """Build ``label``'s model, then measure the RSS high-water-mark increase
    caused specifically by one test-set inference pass (build/quantize/save
    happen *before* the baseline is taken, so they are excluded)."""
    device = torch.device("cpu")  # quantized inference is CPU-only
    model_fp32, cfg = load_checkpoint(checkpoint_path, device)
    data = build_dataloaders(cfg, seed=cfg.seed)
    threshold = cfg.evaluate.threshold

    models_dir = ensure_dir(cfg.paths.models_dir)
    suffix = ".pt" if label == "INT8 static (PTQ)" else ".pth"
    tag = label.replace(" ", "_").replace("(", "").replace(")", "")
    artifact_path = Path(models_dir) / f"_memworker_{cfg.experiment_name}_{tag}{suffix}"

    model = _build_and_save(model_fp32, label, cfg, data, artifact_path)
    size_mb = file_size_mb(artifact_path)
    artifact_path.unlink(missing_ok=True)

    def _inference_pass() -> None:
        collect_predictions(model, data.test_loader, device, threshold)

    _, rss = measure_rss_delta(_inference_pass)
    return {"variant": label, "model_size_mb": round(size_mb, 2), "inference_peak_rss_mb": round(rss, 1)}


def measure_dataset_scenario(cfg_path: str, scenario: str) -> dict[str, float | str]:
    """Measure the RSS high-water-mark increase for one dataset-loading scenario
    (``stream``: iterate the lazy DataLoader once; ``naive``: pre-load the full
    training set into one RAM tensor), isolated in its own process."""
    cfg = load_config(cfg_path)
    root = Path(cfg.data.root)
    samples = scan_split(root / cfg.data.train_dirname)
    transform = build_eval_transform(cfg)
    img = cfg.data.img_size
    n = len(samples)

    if scenario == "stream":
        def _run() -> int:
            ds = ChestXRayDataset(samples, transform, name="train")
            loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
            return sum(imgs.shape[0] for imgs, _ in loader)
    elif scenario == "naive":
        def _run() -> int:
            buf = torch.empty((n, 3, img, img), dtype=torch.float32)
            for i, (path, _) in enumerate(samples):
                with Image.open(path) as im:
                    buf[i] = transform(im.convert("RGB"))
            return n
    else:
        raise ValueError(f"unknown scenario: {scenario!r} (expected one of {SCENARIOS})")

    _, rss = measure_rss_delta(_run)
    return {"scenario": scenario, "n_train_images": n, "peak_rss_mb": round(rss, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated single-sample memory measurement (internal worker process).")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_runtime = sub.add_parser("runtime")
    p_runtime.add_argument("--checkpoint", required=True)
    p_runtime.add_argument("--variant", required=True, choices=list(VARIANTS))

    p_dataset = sub.add_parser("dataset")
    add_config_cli_args(p_dataset)
    p_dataset.add_argument("--scenario", required=True, choices=list(SCENARIOS))

    args = parser.parse_args()
    if args.mode == "runtime":
        result = measure_runtime_variant(args.checkpoint, args.variant)
    else:
        result = measure_dataset_scenario(args.config, args.scenario)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
