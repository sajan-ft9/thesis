#!/usr/bin/env bash
# End-to-end pipeline smoke test on a TINY SYNTHETIC dataset.
#
# Verifies that every stage (validate -> train -> evaluate -> quantize ->
# benchmark -> explain -> report) runs without error and produces its artifacts.
# The numbers it generates are MEANINGLESS (random synthetic data) and exist only
# to prove the code executes. Run with the project venv:  bash scripts/run_smoke_test.sh
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
DATA_DIR="data/synthetic"
CFG="configs/efficientnet_b0.yaml"
# Tiny, fast overrides + point the data root at the synthetic set, CPU-only.
OVR=(--override "data.root=${DATA_DIR}" "train.epochs=2" "train.unfreeze_epoch=2"
     "train.warmup_epochs=1" "train.early_stopping_patience=5" "data.batch_size=8"
     "data.num_workers=0" "evaluate.bootstrap_n=200" "benchmark.warmup=2"
     "benchmark.repeats=5" "quantize.calibration_batches=2" "explain.num_per_category=2"
     "device=cpu")

echo "==> [0/7] Generating synthetic data"
"$PY" scripts/make_synthetic_data.py --out "$DATA_DIR"

echo "==> [1/7] Dataset validation"
"$PY" -m src.dataset --config "$CFG" "${OVR[@]}"

echo "==> [2/7] Training (efficientnet_b0)"
"$PY" -m src.train --config "$CFG" "${OVR[@]}"

echo "==> [2b] Training baselines (resnet18, mobilenetv3)"
"$PY" -m src.train --config configs/resnet18.yaml "${OVR[@]}"
"$PY" -m src.train --config configs/mobilenetv3.yaml "${OVR[@]}"

echo "==> [3/7] Evaluation"
"$PY" -m src.evaluate --checkpoint models/efficientnet_b0_best.pth --device cpu
"$PY" -m src.evaluate --checkpoint models/resnet18_best.pth --device cpu
"$PY" -m src.evaluate --checkpoint models/mobilenetv3_small_best.pth --device cpu

echo "==> [4/7] Benchmarking"
"$PY" -m src.benchmarking --checkpoint models/efficientnet_b0_best.pth --device cpu
"$PY" -m src.benchmarking --checkpoint models/resnet18_best.pth --device cpu
"$PY" -m src.benchmarking --checkpoint models/mobilenetv3_small_best.pth --device cpu

echo "==> [5/7] Quantization study"
"$PY" -m src.quantize --checkpoint models/efficientnet_b0_best.pth

echo "==> [6/7] Grad-CAM++ explainability"
"$PY" -m src.explainability --checkpoint models/efficientnet_b0_best.pth --device cpu

echo "==> [7/7] Reporting (tables + artifacts)"
"$PY" -m src.reporting --config "$CFG" "${OVR[@]}"

echo ""
echo "==> SMOKE TEST COMPLETE. Generated artifacts:"
find results models -type f | sort
echo "(Reminder: synthetic-data numbers are NOT research results.)"
