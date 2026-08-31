"""Memory-footprint analysis for the memory-efficiency thesis claim.

Substantiates the two pillars of the "memory-efficient" contribution with measured,
reproducible peak resident-set-size (RSS) numbers — not `tracemalloc`:

1. **Streaming vs. naive RAM load.** Compares peak RSS when (a) iterating the lazy,
   path-based DataLoader one pass versus (b) pre-loading the entire training set into
   RAM as a single tensor (the common non-streaming anti-pattern). This is the core
   evidence that training/inference is feasible on low-memory hardware.

2. **Runtime memory by precision.** Peak RSS during a full test-set inference pass for
   FP32 vs. dynamic-INT8 vs. static-INT8, alongside on-disk model size, separating the
   model's *weight* footprint from *activation* memory.

Each measurement is a single-purpose subprocess call into ``src.memory_worker``,
repeated several times and aggregated with the **median** (raw samples kept in the
output). This was added after finding that a single peak-RSS sample -- even isolated
per-variant -- is noisy on real hardware (OS scheduling, allocator/page-cache state):
repeating and reporting the spread honestly beats reporting one lucky/unlucky number.

Run:
    python -m src.memory_profile --config configs/efficientnet_b0.yaml \
        --checkpoint models/efficientnet_b0_best.pth --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Config, add_config_cli_args, config_from_cli
from .memory_worker import SCENARIOS, VARIANTS
from .quantize import pick_backend
from .utils import get_logger, save_json

logger = get_logger("memprofile")

__all__ = ["profile_streaming_vs_naive", "profile_runtime_memory", "run_memory_profile"]

DEFAULT_REPEATS = 5


def _run_worker(args: list[str], timeout: int = 1800) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src.memory_worker", *args],
            capture_output=True, text=True, check=True, timeout=timeout,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001 - report, never fabricate
        logger.warning("memory_worker %s failed: %s", args, exc)
        return None


def _median_stats(values: list[float]) -> dict[str, float]:
    return {
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "n": len(values),
    }


def profile_streaming_vs_naive(cfg: Config, config_path: str, repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    """Compare peak RSS: lazy streaming loader vs. full-dataset-in-RAM (float32).

    Each scenario runs ``repeats`` times, each in its own fresh subprocess (see
    module docstring); the headline number is the median, with every raw sample
    kept in the output.
    """
    img = cfg.data.img_size
    from .dataset import scan_split

    n = len(scan_split(Path(cfg.data.root) / cfg.data.train_dirname))
    nominal_float32_mb = n * 3 * img * img * 4 / 1e6
    nominal_uint8_mb = n * 3 * img * img / 1e6

    rss_by_scenario: dict[str, list[float]] = {}
    for scenario in SCENARIOS:
        samples = []
        for _ in range(repeats):
            row = _run_worker(["dataset", "--config", config_path, "--scenario", scenario])
            if row is not None:
                samples.append(row["peak_rss_mb"])
        if samples:
            rss_by_scenario[scenario] = samples
            logger.info("[dataset:%s] peak RSS samples (MB): %s", scenario, samples)

    stream_stats = _median_stats(rss_by_scenario.get("stream", [0.0]))
    naive_stats = _median_stats(rss_by_scenario.get("naive", [0.0]))
    stream_med, naive_med = stream_stats["median"], naive_stats["median"]

    result = {
        "n_train_images": n,
        "img_size": img,
        "streaming_peak_rss_mb": stream_med,
        "streaming_peak_rss_mb_stats": stream_stats,
        "streaming_peak_rss_mb_samples": rss_by_scenario.get("stream", []),
        "naive_full_ram_peak_rss_mb": naive_med,
        "naive_full_ram_peak_rss_mb_stats": naive_stats,
        "naive_full_ram_peak_rss_mb_samples": rss_by_scenario.get("naive", []),
        "naive_nominal_float32_mb": round(nominal_float32_mb, 1),
        "naive_nominal_uint8_mb": round(nominal_uint8_mb, 1),
        "reduction_factor": round(naive_med / stream_med, 1) if stream_med > 0 else None,
    }
    logger.info(
        "Streaming peak RSS median %.0f MB (range %.0f-%.0f) vs naive median %.0f MB (range %.0f-%.0f) (~%.1fx less)",
        stream_stats["median"], stream_stats["min"], stream_stats["max"],
        naive_stats["median"], naive_stats["min"], naive_stats["max"], result["reduction_factor"] or 0,
    )
    return result


def profile_runtime_memory(checkpoint_path: str | Path, cfg: Config, repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    """Peak RSS during test-set inference + on-disk size for FP32 / INT8 variants.

    Each variant runs ``repeats`` times, each in its own fresh subprocess
    (``src.memory_worker``): measuring all three back-to-back in one long-lived
    process lets the allocator reuse pages freed by an earlier variant, biasing
    later readings. The headline RSS is the median across repeats, with every
    raw sample kept in the output.
    """
    backend = pick_backend(cfg.quantize.backend)
    variants: list[dict[str, Any]] = []

    for label in VARIANTS:
        samples: list[dict[str, Any]] = []
        for _ in range(repeats):
            row = _run_worker(["runtime", "--checkpoint", str(checkpoint_path), "--variant", label])
            if row is not None:
                samples.append(row)

        if not samples:
            continue

        rss_values = [s["inference_peak_rss_mb"] for s in samples]
        stats = _median_stats(rss_values)
        row = {
            "variant": label,
            "model_size_mb": samples[0]["model_size_mb"],
            "inference_peak_rss_mb": stats["median"],
            "inference_peak_rss_mb_stats": stats,
            "inference_peak_rss_mb_samples": rss_values,
        }
        variants.append(row)
        logger.info(
            "[%s] weight size %.2f MB | inference peak RSS median %.0f MB (range %.0f-%.0f, n=%d)",
            label, row["model_size_mb"], stats["median"], stats["min"], stats["max"], stats["n"],
        )

    return {"backend": backend, "variants": variants}


def run_memory_profile(
    cfg: Config, checkpoint_path: str | Path | None = None, config_path: str = "configs/efficientnet_b0.yaml",
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Run both memory analyses, save JSON + a bar figure, and return the report."""
    report: dict[str, Any] = {"experiment_name": cfg.experiment_name, "repeats": repeats}
    report["streaming_vs_naive"] = profile_streaming_vs_naive(cfg, config_path, repeats)
    if checkpoint_path:
        report["runtime_memory"] = profile_runtime_memory(checkpoint_path, cfg, repeats)

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

    stream_stats, naive_stats = sv["streaming_peak_rss_mb_stats"], sv["naive_full_ram_peak_rss_mb_stats"]
    medians = [sv["streaming_peak_rss_mb"], sv["naive_full_ram_peak_rss_mb"]]
    err_low = [medians[0] - stream_stats["min"], medians[1] - naive_stats["min"]]
    err_high = [stream_stats["max"] - medians[0], naive_stats["max"] - medians[1]]

    ax = axes[0, 0]
    ax.bar(["Streaming\n(lazy loader)", "Naive\n(full set in RAM)"], medians,
           yerr=[err_low, err_high], capsize=4, color=["#16A34A", "#DC2626"])
    ax.set_ylabel("Peak process RSS (MB), median ± range")
    ax.set_title("Dataset Loading: Memory Footprint")
    for i, v in enumerate(medians):
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
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                         help=f"Repeats per scenario/variant, aggregated by median (default: {DEFAULT_REPEATS}).")
    args = parser.parse_args()
    cfg = config_from_cli(args)
    run_memory_profile(cfg, args.checkpoint, args.config, args.repeats)


if __name__ == "__main__":
    main()
