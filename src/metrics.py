"""Diagnostic metrics and bootstrap confidence intervals.

All metrics are computed from raw ``(y_true, y_prob)`` arrays so the same code
serves training, evaluation and quantization comparisons. Confidence intervals
use non-parametric bootstrap resampling, which makes no distributional
assumptions and is appropriate for the modest test-set size.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "compute_metrics",
    "bootstrap_ci",
    "metric_confidence_intervals",
    "youden_threshold",
    "EPS",
]

EPS = 1e-8


def compute_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full diagnostic metric panel at a given decision threshold.

    Returns accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC and
    the raw confusion-matrix counts (tp/tn/fp/fn).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # Only one class present in y_true (e.g. tiny smoke test) -> AUC undefined.
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def bootstrap_ci(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray | list,
    y_score: np.ndarray | list,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a metric.

    ``metric_func(y_true_resampled, y_score_resampled) -> float``. Returns
    ``(point_estimate, ci_low, ci_high)``. Resamples that raise (e.g. a single
    class drawn) are skipped. Returns NaNs if too few valid resamples.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    try:
        point = float(metric_func(y_true, y_score))
    except ValueError:
        point = float("nan")

    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(float(metric_func(y_true[idx], y_score[idx])))
        except ValueError:
            continue

    if len(scores) < max(50, n_boot // 10):
        return point, float("nan"), float("nan")
    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(low), float(high)


def _accuracy_from_prob(threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    def _fn(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(accuracy_score(y_true, (y_prob >= threshold).astype(int)))

    return _fn


def metric_confidence_intervals(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Bootstrap CIs for the headline metrics (accuracy and ROC-AUC).

    Returns ``{"accuracy": {"point","low","high"}, "auc": {...}}``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    acc_point, acc_low, acc_high = bootstrap_ci(
        _accuracy_from_prob(threshold), y_true, y_prob, n_boot, alpha, seed
    )
    auc_point, auc_low, auc_high = bootstrap_ci(
        roc_auc_score, y_true, y_prob, n_boot, alpha, seed
    )
    return {
        "accuracy": {"point": acc_point, "low": acc_low, "high": acc_high},
        "auc": {"point": auc_point, "low": auc_low, "high": auc_high},
        "n_boot": n_boot,
        "alpha": alpha,
    }


def youden_threshold(y_true: np.ndarray | list, y_prob: np.ndarray | list) -> dict[str, float]:
    """Find the decision threshold maximising Youden's J (sensitivity + specificity - 1)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = int(np.argmax(j))
    return {
        "threshold": float(thresholds[best]),
        "sensitivity": float(tpr[best]),
        "specificity": float(1 - fpr[best]),
        "youden_j": float(j[best]),
    }
