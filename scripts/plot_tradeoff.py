"""Plot the accuracy–efficiency trade-off (the deployment 'frontier' figure).

Reads the existing result JSONs (no recomputation) and produces a two-panel figure:
  (A) ROC-AUC vs model size across architectures (EfficientNet-B0 / ResNet-18 / MobileNetV3),
  (B) ROC-AUC vs peak inference RAM across precisions (FP32 / dynamic-INT8 / static-INT8).

Output: results/figures/accuracy_efficiency_tradeoff.{png,pdf}

Usage:
    python scripts/plot_tradeoff.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.visualization import apply_style, save_figure  # noqa: E402

M = Path("results/metrics")
MODELS = ("efficientnet_b0", "resnet18", "mobilenetv3_small")
LABELS = {"efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet-18", "mobilenetv3_small": "MobileNetV3-Small"}


def _load(name: str) -> dict | None:
    p = M / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: AUC vs model size (architectures).
    for name in MODELS:
        test = _load(f"{name}_test_metrics.json")
        bench = _load(f"{name}_benchmark.json")
        if not (test and bench):
            continue
        auc = test["metrics"]["auc"]
        size = bench[0]["size_mb"]
        axa.scatter(size, auc, s=90)
        axa.annotate(LABELS[name], (size, auc), textcoords="offset points", xytext=(6, 4), fontsize=8)
    axa.set_xlabel("Model size on disk (MB)")
    axa.set_ylabel("ROC-AUC")
    axa.set_title("(A) Accuracy vs size — architectures")

    # Panel B: AUC vs peak inference RAM (precision variants).
    quant = _load("efficientnet_b0_quantization.json")
    mem = _load("memory_profile.json")
    if quant and mem:
        auc_by = {v["variant"]: v["auc"] for v in quant["variants"]}
        rss_by = {v["variant"]: v["inference_peak_rss_mb"] for v in mem["runtime_memory"]["variants"]}
        for variant in ("FP32", "INT8 dynamic", "INT8 static (PTQ)"):
            if variant in auc_by and variant in rss_by:
                axb.scatter(rss_by[variant], auc_by[variant], s=90)
                axb.annotate(variant, (rss_by[variant], auc_by[variant]),
                             textcoords="offset points", xytext=(6, 4), fontsize=8)
    axb.set_xlabel("Peak inference RAM (MB)")
    axb.set_ylabel("ROC-AUC")
    axb.set_title("(B) Accuracy vs memory — quantization")

    fig.suptitle("Accuracy–Efficiency Trade-off (deployment frontier)", fontsize=12, y=1.02)
    fig.tight_layout()
    paths = save_figure(fig, "accuracy_efficiency_tradeoff", "results/figures")
    print(f"[tradeoff] saved {paths['png']}")


if __name__ == "__main__":
    main()
