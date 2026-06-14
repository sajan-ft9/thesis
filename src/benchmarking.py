"""Computational-efficiency benchmarking.

Reports the metrics that actually matter for edge deployment, measured correctly:

* **Model size on disk** — serialised state-dict size in MB.
* **Inference latency** — wall-clock per image at batch size 1, with warmup runs
  discarded and many timed repeats summarised as mean / std / median / p95.
* **Throughput** — images per second at a representative batch size.
* **Peak process RSS** — sampled with ``psutil`` in a background thread during a
  real inference pass.

Deliberately does **not** use ``tracemalloc`` as a memory metric: ``tracemalloc``
only tracks Python-level allocations and grossly understates the true footprint
of tensor/library memory (the source of the indefensible "0.1 MB RAM" figure in
the original work).

Run:
    python -m src.benchmarking --checkpoint models/efficientnet_b0_best.pth
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from .config import Config
from .inference import load_checkpoint
from .utils import get_device, get_logger, save_json

logger = get_logger("benchmark")

__all__ = [
    "model_size_mb",
    "file_size_mb",
    "benchmark_latency",
    "benchmark_throughput",
    "benchmark_peak_rss",
    "benchmark_model",
    "LatencyResult",
]


def model_size_mb(model: torch.nn.Module) -> float:
    """Serialise a model's state-dict to memory and return its size in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


def file_size_mb(path: str | Path) -> float:
    """Return the on-disk size of a file in MB."""
    return Path(path).stat().st_size / 1e6


@dataclass
class LatencyResult:
    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    n: int


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    warmup: int = 10,
    repeats: int = 50,
) -> LatencyResult:
    """Measure single-image inference latency (ms) with warmup + repeated timing."""
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        model(dummy)
    _sync(device)

    timings_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(dummy)
        _sync(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(timings_ms)
    return LatencyResult(
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        n=repeats,
    )


@torch.no_grad()
def benchmark_throughput(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
    input_shape: tuple[int, ...] = (3, 224, 224),
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    """Measure throughput in images/second at a given batch size."""
    model.eval()
    dummy = torch.randn(batch_size, *input_shape, device=device)
    for _ in range(warmup):
        model(dummy)
    _sync(device)

    start = time.perf_counter()
    for _ in range(repeats):
        model(dummy)
    _sync(device)
    elapsed = time.perf_counter() - start
    return float(batch_size * repeats / elapsed)


def benchmark_peak_rss(fn: Callable[[], Any], poll_interval: float = 0.01) -> tuple[Any, float]:
    """Run ``fn`` while sampling process RSS; return ``(result, peak_rss_delta_mb)``."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def _sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, process.memory_info().rss)
            time.sleep(poll_interval)

    thread = threading.Thread(target=_sampler, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    peak = max(peak, process.memory_info().rss)
    return result, (peak - baseline) / 1e6


def benchmark_model(
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device | None = None,
    label: str = "model",
    loader=None,
) -> dict[str, Any]:
    """Run the full benchmark suite on a model and return a flat metrics dict."""
    device = device or get_device(cfg.benchmark.device)
    model = model.to(device).eval()
    img = cfg.data.img_size

    size = model_size_mb(model)
    latency = benchmark_latency(
        model, device, (1, 3, img, img), cfg.benchmark.warmup, cfg.benchmark.repeats
    )
    throughput = benchmark_throughput(
        model, device, cfg.benchmark.throughput_batch_size, (3, img, img)
    )

    peak_rss = float("nan")
    if loader is not None:
        @torch.no_grad()
        def _run_inference() -> None:
            for imgs, _ in loader:
                model(imgs.to(device))

        _, peak_rss = benchmark_peak_rss(_run_inference)

    result = {
        "variant": label,
        "device": str(device),
        "size_mb": round(size, 3),
        "latency_ms_mean": round(latency.mean_ms, 3),
        "latency_ms_std": round(latency.std_ms, 3),
        "latency_ms_median": round(latency.median_ms, 3),
        "latency_ms_p95": round(latency.p95_ms, 3),
        "throughput_img_per_s": round(throughput, 2),
        "peak_rss_delta_mb": round(peak_rss, 2) if peak_rss == peak_rss else None,
        "latency_runs": latency.n,
    }
    logger.info(
        "[%s] size=%.2fMB latency=%.2f±%.2fms throughput=%.1f img/s",
        label, size, latency.mean_ms, latency.std_ms, throughput,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained checkpoint (CPU efficiency).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    result = benchmark_model(model, cfg, device, label=f"{cfg.model.name}_fp32")
    out = Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_benchmark.json"
    save_json([result], out)
    logger.info("Saved benchmark to %s", out)


if __name__ == "__main__":
    main()
