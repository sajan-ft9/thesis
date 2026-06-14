"""Tests for the metric panel and bootstrap confidence intervals."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from src.metrics import (
    bootstrap_ci,
    compute_metrics,
    metric_confidence_intervals,
    youden_threshold,
)


def test_perfect_predictions() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["auc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_confusion_counts_consistent() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_prob = rng.random(50)
    m = compute_metrics(y_true, y_prob)
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 50
    assert m["n"] == 50


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)
    y_prob = np.concatenate([rng.normal(0.3, 0.1, 40), rng.normal(0.7, 0.1, 40)])
    point, low, high = bootstrap_ci(roc_auc_score, y_true, y_prob, n_boot=300, seed=0)
    assert low <= point <= high
    assert not math.isnan(low)


def test_metric_confidence_intervals_structure() -> None:
    rng = np.random.default_rng(2)
    y_true = np.concatenate([np.zeros(30), np.ones(30)]).astype(int)
    y_prob = np.concatenate([rng.random(30) * 0.5, 0.5 + rng.random(30) * 0.5])
    ci = metric_confidence_intervals(y_true, y_prob, n_boot=200)
    assert set(ci["accuracy"]) == {"point", "low", "high"}
    assert set(ci["auc"]) == {"point", "low", "high"}


def test_youden_threshold() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.9]
    out = youden_threshold(y_true, y_prob)
    assert "threshold" in out and 0.0 <= out["sensitivity"] <= 1.0
