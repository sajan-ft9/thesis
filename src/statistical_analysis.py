"""Full statistical characterization for the results chapter (inference-only).

Goes well beyond point metrics, using the *existing* trained checkpoints (no retraining,
no tuning). Produces, from the held-out Kermany test predictions:

* **Bootstrap 95% CIs for ALL metrics** — accuracy, precision, sensitivity, specificity,
  F1, MCC, balanced accuracy, ROC-AUC.
* **Imbalance-robust metrics** — Matthews correlation coefficient (MCC), Cohen's kappa,
  balanced accuracy.
* **Significance tests between models** — McNemar's test (paired errors) and DeLong's test
  (paired AUC), in addition to a paired-bootstrap ΔAUC.
* **Calibration** — ECE, Brier score, calibration slope & intercept, and reliability curves.
* **Threshold robustness** — sensitivity / specificity / F1 across decision thresholds.
* **Error-distribution analysis** — FP vs FN confidence (mean ± std) and their overlap.
* **Effect size** — Cohen's d of the score separation between classes.
* **Dataset baseline** — majority-class accuracy and the model's absolute improvement.

Figures: reliability diagram, threshold-robustness curve, error-probability histogram.

Run:
    python -m src.statistical_analysis
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, norm
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, ChestXRayDataset, build_dataloaders, scan_split
from .inference import collect_predictions, load_checkpoint
from .transforms import build_eval_transform
from .utils import ensure_dir, get_device, get_logger, save_json

logger = get_logger("stats")

MODELS = ("efficientnet_b0", "resnet18", "mobilenetv3_small")
EPS = 1e-8


# --------------------------------------------------------------------------- #
# Point metrics (incl. imbalance-robust)
# --------------------------------------------------------------------------- #
def _metrics_at(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp + EPS)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if (tp + fp) and (tp + fn) and (tn + fp) and (tn + fn) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def bootstrap_full_cis(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5, n_boot: int = 1000, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap 95% CIs for every metric in :func:`_metrics_at`."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = list(_metrics_at(y_true, y_prob, thr).keys())
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        m = _metrics_at(y_true[idx], y_prob[idx], thr)
        for k in keys:
            if not np.isnan(m[k]):
                samples[k].append(m[k])
    point = _metrics_at(y_true, y_prob, thr)
    return {
        k: {
            "point": round(point[k], 4),
            "low": round(float(np.percentile(samples[k], 2.5)), 4),
            "high": round(float(np.percentile(samples[k], 97.5)), 4),
        }
        for k in keys
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    conf = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    correct = ((y_prob >= 0.5).astype(int) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum():
            ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / n
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Logistic recalibration: fit outcome ~ logit(p). slope≈1 & intercept≈0 ⇒ well calibrated."""
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, np.asarray(y_true).astype(int))
    return {"slope": round(float(lr.coef_[0][0]), 4), "intercept": round(float(lr.intercept_[0]), 4)}


def reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {"mean_predicted": [round(float(x), 4) for x in mean_pred],
            "fraction_positive": [round(float(x), 4) for x in frac_pos]}


# --------------------------------------------------------------------------- #
# Significance tests + effect size
# --------------------------------------------------------------------------- #
def paired_auc_bootstrap(y_true, prob_a, prob_b, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx]))
    arr = np.asarray(diffs)
    return {"mean_delta_auc": round(float(arr.mean()), 4),
            "ci_low": round(float(np.percentile(arr, 2.5)), 4),
            "ci_high": round(float(np.percentile(arr, 97.5)), 4),
            "prob_a_gt_b": round(float((arr > 0).mean()), 3)}


def mcnemar_test(y_true, pred_a, pred_b) -> dict[str, float]:
    """Paired test on discordant errors (exact binomial; chi² w/ continuity for reference)."""
    y_true = np.asarray(y_true).astype(int)
    a_ok = (np.asarray(pred_a) == y_true)
    b_ok = (np.asarray(pred_b) == y_true)
    n01 = int((~a_ok & b_ok).sum())  # A wrong, B right
    n10 = int((a_ok & ~b_ok).sum())  # A right, B wrong
    disc = n01 + n10
    p_exact = float(binomtest(min(n01, n10), disc, 0.5).pvalue) if disc else 1.0
    stat = ((abs(n10 - n01) - 1) ** 2) / disc if disc else 0.0
    return {"n01_A_wrong_B_right": n01, "n10_A_right_B_wrong": n10,
            "chi2_cc": round(float(stat), 4), "p_value": round(p_exact, 4)}


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y_true, prob_a, prob_b) -> dict[str, float]:
    """DeLong's paired test for the difference of two correlated ROC-AUCs."""
    y_true = np.asarray(y_true).astype(int)
    order = (-y_true).argsort(kind="mergesort")  # positives (label 1) first
    m = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"delta_auc": round(float(aucs[0] - aucs[1]), 4), "z": round(float(z), 4),
            "p_value": round(float(2 * norm.sf(abs(z))), 4)}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + EPS)
    return float((a.mean() - b.mean()) / sp)


def overlap_coefficient(a, b, bins: int = 20) -> float:
    ha, _ = np.histogram(a, bins=bins, range=(0, 1), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(0, 1), density=True)
    return float(np.minimum(ha, hb).sum() * (1.0 / bins))


def threshold_sweep(y_true, y_prob, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    rows = []
    for t in thresholds:
        m = _metrics_at(y_true, y_prob, t)
        rows.append({"threshold": t, "sensitivity": round(m["sensitivity"], 4),
                     "specificity": round(m["specificity"], 4), "f1": round(m["f1"], 4),
                     "accuracy": round(m["accuracy"], 4)})
    return rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _predict(checkpoint: str, loader, device, thr: float) -> tuple[np.ndarray, np.ndarray]:
    return collect_predictions(load_checkpoint(checkpoint, device)[0], loader, device, thr)[:2]


def run() -> dict:
    device = get_device("auto")
    probs: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    y_true = None
    base_cfg = None
    per_model: dict[str, dict] = {}

    for name in MODELS:
        model, cfg = load_checkpoint(f"models/{name}_best.pth", device)
        base_cfg = base_cfg or cfg
        data = build_dataloaders(cfg, seed=cfg.seed)
        thr = cfg.evaluate.threshold
        yt, yp, _ = collect_predictions(model, data.test_loader, device, thr)
        y_true = yt if y_true is None else y_true
        probs[name] = yp
        preds[name] = (yp >= thr).astype(int)
        m = _metrics_at(yt, yp, thr)
        per_model[name] = {
            "auc": round(m["auc"], 4), "accuracy": round(m["accuracy"], 4),
            "mcc": round(m["mcc"], 4), "cohen_kappa": round(m["cohen_kappa"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "ece": round(expected_calibration_error(yt, yp), 4), "brier": round(brier_score(yt, yp), 4),
            **calibration_slope_intercept(yt, yp),
            "class_separation_cohens_d": round(cohens_d(yp[yt == 1], yp[yt == 0]), 4),
            "reliability": reliability(yt, yp),
            "per_class": _per_class(yt, yp, thr),
        }
        logger.info("[%s] AUC %.4f MCC %.4f kappa %.4f ECE %.4f slope %.3f",
                    name, m["auc"], m["mcc"], m["cohen_kappa"], per_model[name]["ece"], per_model[name]["slope"])

    primary = MODELS[0]
    thr = base_cfg.evaluate.threshold

    # Pairwise significance: bootstrap ΔAUC + DeLong + McNemar.
    pairwise = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            boot = paired_auc_bootstrap(y_true, probs[a], probs[b])
            dl = delong_test(y_true, probs[a], probs[b])
            mc = mcnemar_test(y_true, preds[a], preds[b])
            pairwise.append({"model_a": a, "model_b": b, "bootstrap": boot, "delong": dl, "mcnemar": mc})
            logger.info("%s vs %s | ΔAUC %.4f [%.4f,%.4f] | DeLong p=%.4f | McNemar p=%.4f",
                        a, b, boot["mean_delta_auc"], boot["ci_low"], boot["ci_high"], dl["p_value"], mc["p_value"])

    # Error-distribution analysis (primary model).
    yp = probs[primary]
    pred = preds[primary]
    fp_conf = yp[(pred == 1) & (y_true == 0)]
    fn_conf = yp[(pred == 0) & (y_true == 1)]
    error_dist = {
        "fp_mean": round(float(fp_conf.mean()), 4), "fp_std": round(float(fp_conf.std()), 4), "n_fp": int(len(fp_conf)),
        "fn_mean": round(float(fn_conf.mean()), 4), "fn_std": round(float(fn_conf.std()), 4), "n_fn": int(len(fn_conf)),
        "overlap_coefficient": round(overlap_coefficient(fp_conf, fn_conf), 4),
    }

    # Dataset baseline (majority class) on the test set.
    maj = float(max((y_true == 0).mean(), (y_true == 1).mean()))
    baseline = {"majority_class": CLASS_NAMES[int((y_true == 1).mean() >= 0.5)],
                "baseline_accuracy": round(maj, 4),
                "model_accuracy": round(per_model[primary]["accuracy"], 4),
                "improvement_pp": round((per_model[primary]["accuracy"] - maj) * 100, 1)}

    report = {
        "dataset_baseline": baseline,
        "primary_full_cis": bootstrap_full_cis(y_true, probs[primary], thr),
        "per_model_kermany": per_model,
        "pairwise_kermany": pairwise,
        "threshold_sweep_primary": threshold_sweep(y_true, probs[primary]),
        "error_distribution_primary": error_dist,
    }

    # External RSNA calibration (primary model), if available.
    import os
    rsna = "data/processed/rsna_external"
    if os.path.isdir(rsna):
        samples = scan_split(rsna)
        ds = ChestXRayDataset(samples, build_eval_transform(base_cfg), name="rsna")
        loader = DataLoader(ds, batch_size=base_cfg.data.batch_size, shuffle=False, num_workers=base_cfg.data.num_workers)
        yt, yp = _predict(f"models/{primary}_best.pth", loader, device, thr)
        report["external_rsna_calibration"] = {
            "ece": round(expected_calibration_error(yt, yp), 4), "brier": round(brier_score(yt, yp), 4),
            **calibration_slope_intercept(yt, yp),
        }

    save_json(report, ensure_dir("results/metrics") / "statistical_analysis.json")
    _figures(report, per_model)
    logger.info("Saved statistical analysis + figures")
    return report


def _per_class(y_true, y_prob, threshold: float) -> dict:
    rep = classification_report(y_true, (y_prob >= threshold).astype(int), target_names=list(CLASS_NAMES),
                                output_dict=True, zero_division=0)
    return {c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score", "support")} for c in CLASS_NAMES}


def _figures(report: dict, per_model: dict) -> None:
    import matplotlib.pyplot as plt

    from .visualization import apply_style, save_figure
    apply_style()
    out = "results/figures"

    # Reliability diagram (all models).
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
    for name in MODELS:
        r = per_model[name]["reliability"]
        ax.plot(r["mean_predicted"], r["fraction_positive"], "o-", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "calibration_reliability", out)

    # Threshold robustness (primary).
    sweep = report["threshold_sweep_primary"]
    t = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for key, lab in (("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("f1", "F1")):
        ax.plot(t, [r[key] for r in sweep], "o-", label=lab)
    ax.axvline(0.5, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Robustness (EfficientNet-B0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "threshold_robustness", out)

    # Error-probability histogram (primary).
    ed = report["error_distribution_primary"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.axvspan(0, 0.5, color="#FEE2E2", alpha=0.4)
    ax.set_title(f"Error confidence (FP n={ed['n_fp']}, FN n={ed['n_fn']}; overlap={ed['overlap_coefficient']})")
    ax.set_xlabel("Predicted pneumonia probability")
    ax.set_ylabel("Count")
    ax.axvline(ed["fp_mean"], color="#DC2626", ls="--", label=f"FP mean {ed['fp_mean']:.2f}")
    ax.axvline(ed["fn_mean"], color="#2563EB", ls="--", label=f"FN mean {ed['fn_mean']:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "error_distribution", out)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
