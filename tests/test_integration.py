"""End-to-end integration: train -> evaluate -> benchmark on synthetic data.

Exercises the full in-process pipeline (one tiny epoch) to guarantee the stages
compose correctly and produce their artifacts. Numbers are meaningless (synthetic
data); only successful execution and file creation are asserted.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.benchmarking import benchmark_model
from src.evaluate import evaluate_checkpoint
from src.inference import load_checkpoint
from src.train import train


def test_train_evaluate_benchmark(cfg, tmp_path: Path) -> None:
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.paths.results_dir = str(tmp_path / "results")

    # Train (1 epoch).
    result = train(cfg)
    ckpt = Path(result["checkpoint"])
    assert ckpt.exists()
    assert (Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_history.csv").exists()
    assert (Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_metadata.json").exists()

    # Evaluate.
    summary = evaluate_checkpoint(str(ckpt), cfg=cfg, device_str="cpu")
    assert summary["n_test"] > 0
    assert (Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_test_metrics.json").exists()
    assert Path(summary["figures"]["roc_curve"]["png"]).exists()

    # Benchmark.
    model, loaded_cfg = load_checkpoint(str(ckpt), torch.device("cpu"))
    bench = benchmark_model(model, loaded_cfg, torch.device("cpu"), label="fp32")
    assert bench["size_mb"] > 0
    assert bench["latency_ms_mean"] > 0
