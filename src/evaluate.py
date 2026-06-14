"""Evaluation entry point.

Loads a trained checkpoint, evaluates it on the held-out test set, and produces:
the full diagnostic metric panel, bootstrap 95% confidence intervals for accuracy
and ROC-AUC, the Youden-J optimal operating threshold, confusion-matrix / ROC /
PR figures, and a misclassification (error analysis) table. All numeric outputs
are written to ``results/metrics/`` as JSON for downstream report rendering.

Run:
    python -m src.evaluate --checkpoint models/efficientnet_b0_best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .config import Config
from .dataset import CLASS_NAMES, build_dataloaders
from .inference import collect_predictions, load_checkpoint
from .metrics import compute_metrics, metric_confidence_intervals, youden_threshold
from .utils import ensure_dir, get_device, get_logger, save_json

logger = get_logger("evaluate")

__all__ = ["evaluate_checkpoint", "error_analysis"]


def error_analysis(
    test_samples: list[tuple[str, int]], y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Build a per-error table (FP/FN) with confidence for qualitative review."""
    y_pred = (y_prob >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for (path, _), true, prob, pred in zip(test_samples, y_true, y_prob, y_pred):
        if pred != true:
            rows.append(
                {
                    "path": path,
                    "true_label": CLASS_NAMES[int(true)],
                    "pred_label": CLASS_NAMES[int(pred)],
                    "probability": round(float(prob), 4),
                    "error_type": "FP" if pred == 1 else "FN",
                    "confidence_margin": round(abs(float(prob) - 0.5), 4),
                }
            )
    return pd.DataFrame(rows)


def evaluate_checkpoint(
    checkpoint_path: str | Path, cfg: Config | None = None, device_str: str = "auto"
) -> dict[str, Any]:
    """Evaluate a checkpoint on the test set and persist metrics + figures."""
    device = get_device(device_str)
    model, ckpt_cfg = load_checkpoint(checkpoint_path, device)
    cfg = cfg or ckpt_cfg

    data = build_dataloaders(cfg, seed=cfg.seed)
    y_true, y_prob, _ = collect_predictions(model, data.test_loader, device, cfg.evaluate.threshold)

    metrics = compute_metrics(y_true, y_prob, cfg.evaluate.threshold)
    cis = metric_confidence_intervals(
        y_true, y_prob, cfg.evaluate.bootstrap_n, cfg.evaluate.bootstrap_alpha, cfg.seed, cfg.evaluate.threshold
    )
    youden = youden_threshold(y_true, y_prob)

    results_dir = Path(cfg.paths.results_dir)
    metrics_dir = ensure_dir(results_dir / "metrics")
    name = cfg.experiment_name

    # Figures (lazy import to keep evaluate importable without a display).
    from .visualization import plot_confusion_matrix, plot_pr_curve, plot_roc_curve

    cm = confusion_matrix(y_true, (y_prob >= cfg.evaluate.threshold).astype(int), labels=[0, 1])
    fig_cm = plot_confusion_matrix(cm, CLASS_NAMES, results_dir / "confusion_matrices", stem=f"{name}_confusion_matrix")
    fig_roc = plot_roc_curve(y_true, y_prob, results_dir / "roc_curves", stem=f"{name}_roc_curve")
    fig_pr = plot_pr_curve(y_true, y_prob, results_dir / "figures", stem=f"{name}_pr_curve")

    # Error analysis.
    err_df = error_analysis(data.test_samples, y_true, y_prob, cfg.evaluate.threshold)
    err_csv = metrics_dir / f"{name}_error_cases.csv"
    err_df.to_csv(err_csv, index=False)

    report_txt = classification_report(
        y_true, (y_prob >= cfg.evaluate.threshold).astype(int), target_names=list(CLASS_NAMES), zero_division=0
    )

    summary = {
        "experiment_name": name,
        "checkpoint": str(checkpoint_path),
        "n_test": int(len(y_true)),
        "metrics": metrics,
        "confidence_intervals": cis,
        "youden_threshold": youden,
        "n_errors": int(len(err_df)),
        "n_false_positive": int((err_df["error_type"] == "FP").sum()) if len(err_df) else 0,
        "n_false_negative": int((err_df["error_type"] == "FN").sum()) if len(err_df) else 0,
        "figures": {"confusion_matrix": fig_cm, "roc_curve": fig_roc, "pr_curve": fig_pr},
        "error_cases_csv": str(err_csv),
    }
    save_json(summary, metrics_dir / f"{name}_test_metrics.json")

    logger.info("Test metrics for %s:", name)
    logger.info(
        "  AUC=%.4f (95%% CI %.4f-%.4f) | Acc=%.4f | Sens=%.4f | Spec=%.4f | F1=%.4f",
        metrics["auc"], cis["auc"]["low"], cis["auc"]["high"],
        metrics["accuracy"], metrics["sensitivity"], metrics["specificity"], metrics["f1"],
    )
    logger.info("\n%s", report_txt)
    logger.info("Errors: %d (FP=%d, FN=%d)", summary["n_errors"], summary["n_false_positive"], summary["n_false_negative"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set.")
    parser.add_argument("--checkpoint", required=True, help="Path to a *_best.pth checkpoint.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, device_str=args.device)


if __name__ == "__main__":
    main()
