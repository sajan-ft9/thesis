"""Publication-quality figure generation.

A single shared matplotlib style is applied so every figure in the thesis/paper
looks consistent. Each plotting function returns the saved file paths; figures
are exported as **both PNG (300 dpi) and PDF (vector)** for print-quality use.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc as sk_auc,
)
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

from .utils import ensure_dir  # noqa: E402

__all__ = [
    "apply_style",
    "save_figure",
    "plot_class_distribution",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_quantization_comparison",
    "plot_model_size_comparison",
]

_PALETTE = {
    "primary": "#2563EB",
    "accent": "#DC2626",
    "muted": "#64748B",
    "good": "#16A34A",
}


def apply_style() -> None:
    """Apply a consistent, paper-friendly matplotlib style."""
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str, out_dir: str | Path) -> dict[str, str]:
    """Save a figure as PNG + PDF under ``out_dir``; return ``{"png":..., "pdf":...}``."""
    out_dir = ensure_dir(out_dir)
    paths: dict[str, str] = {}
    for ext in ("png", "pdf"):
        path = Path(out_dir) / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def plot_class_distribution(per_split_rows: list[dict[str, Any]], out_dir: str | Path) -> dict[str, str]:
    """Grouped bar chart of Normal/Pneumonia counts per split (excludes 'total')."""
    apply_style()
    rows = [r for r in per_split_rows if r["split"] != "total"]
    splits = [r["split"] for r in rows]
    normal = [r["normal"] for r in rows]
    pneumonia = [r["pneumonia"] for r in rows]
    x = np.arange(len(splits))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width / 2, normal, width, label="Normal", color=_PALETTE["primary"])
    ax.bar(x + width / 2, pneumonia, width, label="Pneumonia", color=_PALETTE["accent"])
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of images")
    ax.set_title("Class Distribution per Split")
    ax.legend()
    for i, (n, p) in enumerate(zip(normal, pneumonia)):
        ax.text(i - width / 2, n, str(n), ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, p, str(p), ha="center", va="bottom", fontsize=8)
    return save_figure(fig, "dataset_distribution", out_dir)


def plot_training_curves(history: list[dict[str, Any]], out_dir: str | Path) -> dict[str, str]:
    """Plot loss, AUC and learning-rate curves over epochs."""
    apply_style()
    epochs = [r["epoch"] for r in history]

    def col(name: str) -> list[float]:
        return [r.get(name, float("nan")) for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, col("train_loss"), label="Train", color=_PALETTE["primary"])
    axes[0].plot(epochs, col("val_loss"), label="Val", color=_PALETTE["accent"])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, col("train_auc"), label="Train", color=_PALETTE["primary"])
    axes[1].plot(epochs, col("val_auc"), label="Val", color=_PALETTE["accent"])
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, col("lr"), color=_PALETTE["muted"])
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    fig.tight_layout()
    return save_figure(fig, "training_curves", out_dir)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: Sequence[str], out_dir: str | Path, stem: str = "confusion_matrix"
) -> dict[str, str]:
    """Annotated confusion-matrix heatmap."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return save_figure(fig, stem, out_dir)


def plot_roc_curve(
    y_true: np.ndarray | list, y_prob: np.ndarray | list, out_dir: str | Path, stem: str = "roc_curve"
) -> dict[str, str]:
    """ROC curve with AUC annotation."""
    apply_style()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=_PALETTE["primary"], lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, stem, out_dir)


def plot_pr_curve(
    y_true: np.ndarray | list, y_prob: np.ndarray | list, out_dir: str | Path, stem: str = "pr_curve"
) -> dict[str, str]:
    """Precision-Recall curve with average-precision annotation."""
    apply_style()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = sk_auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, color=_PALETTE["good"], lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save_figure(fig, stem, out_dir)


def plot_quantization_comparison(rows: list[dict[str, Any]], out_dir: str | Path) -> dict[str, str]:
    """Bar charts comparing model variants on size, latency and AUC."""
    apply_style()
    labels = [r["variant"] for r in rows]
    colors = [_PALETTE["primary"], _PALETTE["accent"], _PALETTE["good"]][: len(rows)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(
        axes,
        ["size_mb", "latency_ms_mean", "auc"],
        ["Model Size (MB)", "Latency (ms/img)", "ROC-AUC"],
    ):
        values = [r.get(key, float("nan")) for r in rows]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(values):
            if v == v:  # not NaN
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "quantization_comparison", out_dir)


def plot_model_size_comparison(rows: list[dict[str, Any]], out_dir: str | Path) -> dict[str, str]:
    """Horizontal bar chart of disk size across model variants."""
    apply_style()
    labels = [r["variant"] for r in rows]
    sizes = [r.get("size_mb", float("nan")) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(rows) + 1.5))
    ax.barh(labels, sizes, color=_PALETTE["primary"])
    ax.set_xlabel("Model size on disk (MB)")
    ax.set_title("Model Size Comparison")
    for i, v in enumerate(sizes):
        if v == v:
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "model_size_comparison", out_dir)
