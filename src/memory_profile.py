"""Memory-footprint analysis for the memory-efficiency thesis claim.

Substantiates the two pillars of the "memory-efficient" contribution with measured,
reproducible peak resident-set-size (RSS) numbers — not `tracemalloc`:

1. **Streaming vs. naive RAM load.** Compares peak process RSS when (a) iterating the
   lazy, path-based DataLoader one pass versus (b) pre-loading the entire training set
   into RAM as a single tensor (the common non-streaming anti-pattern). This is the
   core evidence that training/inference is feasible on low-memory hardware.

2. **Runtime memory by precision.** Peak RSS during a full test-set inference pass for
   FP32 vs. dynamic-INT8 vs. static-INT8, alongside on-disk model size, separating the
   model's *weight* footprint from *activation* memory.

Run:
    python -m src.memory_profile --config configs/efficientnet_b0.yaml \
        --checkpoint models/efficientnet_b0_best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .benchmarking import benchmark_peak_rss, model_size_mb
from .config import Config, add_config_cli_args, config_from_cli
from .dataset import ChestXRayDataset, scan_split
from .inference import collect_predictions, load_checkpoint
from .quantize import dynamic_quantize, pick_backend, static_quantize
from .transforms import build_eval_transform
from .utils import get_logger, save_json

logger = get_logger("memprofile")

__all__ = ["profile_streaming_vs_naive", "profile_runtime_memory", "run_memory_profile"]


def profile_streaming_vs_naive(cfg: Config) -> dict[str, Any]:
    """Compare peak RSS: lazy streaming loader vs. full-dataset-in-RAM (float32)."""
    root = Path(cfg.data.root)
    samples = scan_split(root / cfg.data.train_dirname)
    transform = build_eval_transform(cfg)
    img = cfg.data.img_size
    n = len(samples)
    nominal_float32_mb = n * 3 * img * img * 4 / 1e6
    nominal_uint8_mb = n * 3 * img * img / 1e6

    # (a) Streaming: iterate the lazy DataLoader once (num_workers=0 so all memory is
    #     in this process and directly comparable to the naive baseline).
    def _stream() -> int:
        ds = ChestXRayDataset(samples, transform, name="train")
        loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
        seen = 0
        for imgs, _ in loader:
            seen += imgs.shape[0]
        return seen

    _, stream_rss = benchmark_peak_rss(_stream)

    # (b) Naive: pre-load the entire training set into one RAM tensor, then iterate it.
    def _naive() -> int:
        buf = torch.empty((n, 3, img, img), dtype=torch.float32)
        for i, (path, _) in enumerate(samples):
            with Image.open(path) as im:
                buf[i] = transform(im.convert("RGB"))
        seen = 0
        for start in range(0, n, cfg.data.batch_size):
            seen += buf[start : start + cfg.data.batch_size].shape[0]
        return seen

    _, naive_rss = benchmark_peak_rss(_naive)

    result = {
        "n_train_images": n,
        "img_size": img,
        "streaming_peak_rss_mb": round(stream_rss, 1),
        "naive_full_ram_peak_rss_mb": round(naive_rss, 1),
        "naive_nominal_float32_mb": round(nominal_float32_mb, 1),
        "naive_nominal_uint8_mb": round(nominal_uint8_mb, 1),
        "reduction_factor": round(naive_rss / stream_rss, 1) if stream_rss > 0 else None,
    }
    logger.info(
        "Streaming peak RSS %.0f MB vs naive full-RAM %.0f MB (~%.0fx less)",
        stream_rss, naive_rss, result["reduction_factor"] or 0,
    )
    return result


def _inference_pass(model: torch.nn.Module, loader, threshold: float) -> None:
    collect_predictions(model, loader, torch.device("cpu"), threshold)


def profile_runtime_memory(checkpoint_path: str | Path, cfg: Config) -> dict[str, Any]:
    """Peak RSS during test-set inference + on-disk size for FP32 / INT8 variants."""
    from .dataset import build_dataloaders

    device = torch.device("cpu")  # quantized inference is CPU-only; compare on equal footing
    model_fp32, ckpt_cfg = load_checkpoint(checkpoint_path, device)
    cfg = cfg or ckpt_cfg
    data = build_dataloaders(cfg, seed=cfg.seed)
    backend = pick_backend(cfg.quantize.backend)
    thr = cfg.evaluate.threshold

    variants: list[dict[str, Any]] = []

    def record(model: torch.nn.Module, label: str) -> None:
        _, rss = benchmark_peak_rss(lambda: _inference_pass(model, data.test_loader, thr))
        variants.append(
            {"variant": label, "model_size_mb": round(model_size_mb(model), 2), "inference_peak_rss_mb": round(rss, 1)}
        )
        logger.info("[%s] weight size %.2f MB | inference peak RSS %.0f MB", label, variants[-1]["model_size_mb"], rss)

    record(model_fp32, "FP32")
    try:
        record(dynamic_quantize(model_fp32, cfg.quantize.backend), "INT8 dynamic")
    except Exception as exc:  # noqa: BLE001
        logger.warning("dynamic quant skipped: %s", exc)
    try:
        stat = static_quantize(model_fp32, data.val_loader, backend, cfg.data.img_size, cfg.quantize.calibration_batches)
        record(stat, "INT8 static (PTQ)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("static PTQ skipped: %s", exc)

    return {"backend": backend, "variants": variants}


def run_memory_profile(cfg: Config, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Run both memory analyses, save JSON + a bar figure, and return the report."""
    report: dict[str, Any] = {"experiment_name": cfg.experiment_name}
    report["streaming_vs_naive"] = profile_streaming_vs_naive(cfg)
    if checkpoint_path:
        report["runtime_memory"] = profile_runtime_memory(checkpoint_path, cfg)

    out = Path(cfg.paths.results_dir) / "metrics" / "memory_profile.json"
    save_json(report, out)
    _plot(report, Path(cfg.paths.results_dir) / "figures")
    logger.info("Memory profile saved to %s", out)
    return report


def _plot(report: dict[str, Any], out_dir: str | Path) -> None:
    """Two-panel bar chart: dataset loading strategy + per-precision footprint."""
    import matplotlib.pyplot as plt

    from .visualization import apply_style, save_figure

    apply_style()
    sv = report["streaming_vs_naive"]
    rt = report.get("runtime_memory", {}).get("variants", [])
    fig, axes = plt.subplots(1, 2 if rt else 1, figsize=(12 if rt else 6, 4.2), squeeze=False)

    ax = axes[0, 0]
    ax.bar(["Streaming\n(lazy loader)", "Naive\n(full set in RAM)"],
           [sv["streaming_peak_rss_mb"], sv["naive_full_ram_peak_rss_mb"]],
           color=["#16A34A", "#DC2626"])
    ax.set_ylabel("Peak process RSS (MB)")
    ax.set_title("Dataset Loading: Memory Footprint")
    for i, v in enumerate([sv["streaming_peak_rss_mb"], sv["naive_full_ram_peak_rss_mb"]]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    if rt:
        ax2 = axes[0, 1]
        labels = [v["variant"] for v in rt]
        ax2.bar(labels, [v["model_size_mb"] for v in rt], color="#2563EB")
        ax2.set_ylabel("Model size on disk (MB)")
        ax2.set_title("Model Footprint by Precision")
        ax2.tick_params(axis="x", rotation=15)
        for i, v in enumerate(rt):
            ax2.text(i, v["model_size_mb"], f"{v['model_size_mb']:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_figure(fig, "memory_footprint", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-footprint analysis (streaming vs naive; per-precision).")
    add_config_cli_args(parser)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for per-precision runtime memory.")
    args = parser.parse_args()
    cfg = config_from_cli(args)
    run_memory_profile(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
