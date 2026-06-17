"""Deeper statistical analysis for the results chapter (inference-only).

Adds rigour beyond point metrics, using the *existing* trained checkpoints (no
retraining, no tuning):

* **Paired bootstrap AUC comparison** between models on the same test set with the same
  resample indices — gives the mean ΔAUC, its 95% CI, and P(model A > model B), the
  honest way to say whether two models differ.
* **Calibration**: Expected Calibration Error (ECE) and Brier score — does the predicted
  probability mean what it says?
* **Per-class breakdown**: precision / recall / F1 / support for Normal and Pneumonia.

Run:
    python -m src.statistical_analysis
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, ChestXRayDataset, build_dataloaders, scan_split
from .inference import collect_predictions, load_checkpoint
from .metrics import compute_metrics
from .transforms import build_eval_transform
from .utils import ensure_dir, get_device, get_logger, save_json

logger = get_logger("stats")

MODELS = ("efficientnet_b0", "resnet18", "mobilenetv3_small")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: gap between confidence and accuracy, weighted by bin population."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    conf = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    correct = ((y_prob >= 0.5).astype(int) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / n
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and the binary label."""
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def paired_auc_bootstrap(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Paired bootstrap of (AUC_a − AUC_b) using shared resample indices."""
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        diffs.append(roc_auc_score(yt, prob_a[idx]) - roc_auc_score(yt, prob_b[idx]))
    arr = np.asarray(diffs)
    return {
        "mean_delta_auc": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "prob_a_gt_b": float((arr > 0).mean()),
        "n_resamples": int(len(arr)),
    }


def _per_class(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    rep = classification_report(
        y_true, (y_prob >= threshold).astype(int), target_names=list(CLASS_NAMES),
        output_dict=True, zero_division=0,
    )
    return {c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score", "support")} for c in CLASS_NAMES}


def _predict_dir(checkpoint: str, image_dir: str, cfg, device) -> tuple[np.ndarray, np.ndarray]:
    samples = scan_split(image_dir)
    ds = ChestXRayDataset(samples, build_eval_transform(cfg), name="ext")
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    yt, yp, _ = collect_predictions(load_checkpoint(checkpoint, device)[0], loader, device, cfg.evaluate.threshold)
    return yt, yp


def run() -> dict:
    device = get_device("auto")
    # Collect Kermany-test predictions for all three models (shared, deterministic order).
    probs: dict[str, np.ndarray] = {}
    y_true = None
    base_cfg = None
    per_model: dict[str, dict] = {}
    for name in MODELS:
        model, cfg = load_checkpoint(f"models/{name}_best.pth", device)
        base_cfg = base_cfg or cfg
        data = build_dataloaders(cfg, seed=cfg.seed)
        yt, yp, _ = collect_predictions(model, data.test_loader, device, cfg.evaluate.threshold)
        y_true = yt if y_true is None else y_true
        probs[name] = yp
        m = compute_metrics(yt, yp, cfg.evaluate.threshold)
        per_model[name] = {
            "auc": round(m["auc"], 4),
            "accuracy": round(m["accuracy"], 4),
            "ece": round(expected_calibration_error(yt, yp), 4),
            "brier": round(brier_score(yt, yp), 4),
            "per_class": _per_class(yt, yp, cfg.evaluate.threshold),
        }
        logger.info("[%s] AUC %.4f | ECE %.4f | Brier %.4f", name, m["auc"], per_model[name]["ece"], per_model[name]["brier"])

    # Pairwise paired-bootstrap AUC comparisons on Kermany test.
    pairwise = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            r = paired_auc_bootstrap(y_true, probs[a], probs[b])
            r.update({"model_a": a, "model_b": b})
            sig = "significant" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "not significant (CI spans 0)"
            r["interpretation"] = f"ΔAUC(A−B)={r['mean_delta_auc']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] — {sig}"
            pairwise.append(r)
            logger.info("%s vs %s: %s", a, b, r["interpretation"])

    report = {"per_model_kermany": per_model, "pairwise_auc_kermany": pairwise}

    # Calibration of the primary model on the external RSNA set, if present.
    import os

    rsna_dir = "data/processed/rsna_external"
    if os.path.isdir(rsna_dir):
        yt, yp = _predict_dir("models/efficientnet_b0_best.pth", rsna_dir, base_cfg, device)
        report["external_rsna_calibration"] = {
            "ece": round(expected_calibration_error(yt, yp), 4),
            "brier": round(brier_score(yt, yp), 4),
        }
        logger.info("RSNA calibration: ECE %.4f Brier %.4f",
                    report["external_rsna_calibration"]["ece"], report["external_rsna_calibration"]["brier"])

    out = ensure_dir("results/metrics") / "statistical_analysis.json"
    save_json(report, out)
    logger.info("Saved statistical analysis to %s", out)
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
