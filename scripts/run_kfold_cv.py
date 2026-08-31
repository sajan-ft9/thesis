"""Stratified k-fold cross-validation for the primary model (EfficientNet-B0).

Addresses the thesis's own acknowledged limitation ("single stratified split +
bootstrap CIs, no k-fold cross-validation") by training the primary model on
``folds`` independent stratified partitions of the training pool, and evaluating
*every* fold's checkpoint on the same untouched official Kermany test set
(``data/raw/chest_xray/test``, n=624) -- never used for fold assignment, training,
or early stopping in any fold. Aggregating those ``folds`` independent test-set
scores (mean +/- std) is a stronger evidence of generalisation than one split's
bootstrap CI, because the *training data itself* varies across folds, not just
the resampling of a fixed test set.

Each fold is a full, independent training run (same hyperparameters/early
stopping as the main pipeline) and takes as long as one normal training run --
budget accordingly (see ``--folds`` and the module-level timing note in
reports/thesis_final.md).

Run:
    python scripts/run_kfold_cv.py --config configs/efficientnet_b0.yaml --folds 5
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import StratifiedKFold

from src.config import Config, add_config_cli_args, config_from_cli
from src.dataset import build_dataloaders_from_samples, deduplicate_samples, scan_split
from src.evaluate import evaluate_checkpoint
from src.train import train
from src.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("kfold")


def run_kfold(cfg: Config, folds: int, cv_seed: int) -> dict[str, Any]:
    root = Path(cfg.data.root)
    all_train = scan_split(root / cfg.data.train_dirname)
    test_samples = scan_split(root / cfg.data.test_dirname)
    if cfg.data.deduplicate:
        all_train, removed = deduplicate_samples(all_train)
        if removed:
            logger.info("Deduplicated train pool: removed %d byte-identical duplicate image(s)", len(removed))

    labels = [label for _, label in all_train]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cv_seed)

    base_name = cfg.experiment_name
    fold_results: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_train, labels)):
        fold_name = f"{base_name}_kfold{fold_idx}"
        logger.info("=== Fold %d/%d (%s) ===", fold_idx + 1, folds, fold_name)

        train_samples = [all_train[i] for i in train_idx]
        val_samples = [all_train[i] for i in val_idx]
        fold_seed = cfg.seed + fold_idx

        cfg_fold = copy.deepcopy(cfg)
        cfg_fold.experiment_name = fold_name
        cfg_fold.seed = fold_seed

        data = build_dataloaders_from_samples(cfg_fold, train_samples, val_samples, test_samples, seed=fold_seed)
        logger.info("Fold %d split sizes -> train=%d val=%d test=%d", fold_idx, len(train_samples), len(val_samples), len(test_samples))

        train_summary = train(cfg_fold, data=data)
        metadata = load_json(Path(cfg_fold.paths.results_dir) / "metrics" / f"{fold_name}_metadata.json")
        best_val_score = metadata["best_val_score"]
        test_summary = evaluate_checkpoint(train_summary["checkpoint"], cfg=cfg_fold, device_str="auto")

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "best_epoch": train_summary["best_epoch"],
            "val_auc": best_val_score,
            "test_metrics": test_summary["metrics"],
        })
        logger.info(
            "Fold %d done: best_epoch=%d val_auc=%.4f test_auc=%.4f test_acc=%.4f",
            fold_idx, train_summary["best_epoch"], best_val_score,
            test_summary["metrics"]["auc"], test_summary["metrics"]["accuracy"],
        )

    return {"folds": folds, "cv_seed": cv_seed, "base_experiment": base_name, "fold_results": fold_results}


def _agg(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {}
    for name in metric_names:
        values = [fr["test_metrics"][name] for fr in report["fold_results"]]
        summary[name] = _agg(values)
    report["test_metric_summary"] = summary
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified k-fold CV for the primary model.")
    add_config_cli_args(parser)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=None, help="Defaults to the config's seed.")
    parser.add_argument("--out", default=None, help="Defaults to results/metrics/<experiment_name>_kfold.json")
    args = parser.parse_args()

    cfg = config_from_cli(args)
    cv_seed = args.cv_seed if args.cv_seed is not None else cfg.seed

    report = run_kfold(cfg, args.folds, cv_seed)
    report = summarize(report)

    out = Path(args.out) if args.out else Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_kfold.json"
    ensure_dir(out.parent)
    save_json(report, out)

    logger.info("K-fold CV summary (n=%d folds):", args.folds)
    for name, stats in report["test_metric_summary"].items():
        logger.info("  %-12s mean=%.4f std=%.4f range=[%.4f, %.4f]", name, stats["mean"], stats["std"], stats["min"], stats["max"])
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
