#!/usr/bin/env bash
# Full REAL pipeline on the Kermany dataset (Apple MPS GPU).
# Mirrors `make reproduce` but pins the device to MPS and enables the MPS CPU
# fallback for any op not yet implemented on Metal. Quantization/benchmark run on
# CPU (INT8 inference is CPU-only) as configured.
#
# Usage:  bash scripts/run_full_pipeline.sh [EPOCHS]
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
EPOCHS="${1:-20}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

DEV=(--override "device=mps" "train.epochs=${EPOCHS}")

echo "==> [1/7] Dataset validation"
$PY -m src.dataset --config configs/efficientnet_b0.yaml

echo "==> [2/7] Training (device=mps, epochs=${EPOCHS})"
$PY -m src.train --config configs/efficientnet_b0.yaml "${DEV[@]}"
$PY -m src.train --config configs/resnet18.yaml "${DEV[@]}"
$PY -m src.train --config configs/mobilenetv3.yaml "${DEV[@]}"

echo "==> [3/7] Evaluation (test set + bootstrap CIs + figures)"
$PY -m src.evaluate --checkpoint models/efficientnet_b0_best.pth --device mps
$PY -m src.evaluate --checkpoint models/resnet18_best.pth --device mps
$PY -m src.evaluate --checkpoint models/mobilenetv3_small_best.pth --device mps

echo "==> [4/7] Efficiency benchmarking (CPU)"
$PY -m src.benchmarking --checkpoint models/efficientnet_b0_best.pth --device cpu
$PY -m src.benchmarking --checkpoint models/resnet18_best.pth --device cpu
$PY -m src.benchmarking --checkpoint models/mobilenetv3_small_best.pth --device cpu

echo "==> [5/7] Quantization study (dynamic + static PTQ + ONNX)"
$PY -m src.quantize --checkpoint models/efficientnet_b0_best.pth

echo "==> [6/7] Grad-CAM++ explanations (CPU for stable gradients)"
$PY -m src.explainability --checkpoint models/efficientnet_b0_best.pth --device cpu

echo "==> [7/7] Reporting + thesis rendering"
$PY -m src.reporting --config configs/efficientnet_b0.yaml
$PY scripts/render_report.py --config configs/efficientnet_b0.yaml

echo ""
echo "==> FULL PIPELINE COMPLETE."
