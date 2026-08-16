# Controlled publication benchmark record

This file identifies the one measured run whose machine-dependent efficiency values
should be used in the manuscript. Classification values are unchanged across the
rerun; timing and process-RSS values are tied to this Docker/CPU environment.

**Run date:** 2026-08-16  
**Command:** `make docker-actual`  
**Verification:** `make docker-verify` — 26 stable checks passed  
**Software:** Docker CPU image, PyTorch 2.12.0, QNNPACK backend for quantization  
**Dataset:** real Kermany pediatric CXR data; no synthetic data and no retraining in
this benchmark record  
**Primary checkpoint:** `models/efficientnet_b0_best.pth`

## Architecture benchmark

This is the single-image, warm-up-corrected benchmark from
`results/metrics/*_benchmark.json`:

| Model | Mean latency (ms/image) | P95 (ms) | Throughput (image/s) |
|---|---:|---:|---:|
| EfficientNet-B0 FP32 | 92.821 | 131.879 | 41.71 |
| ResNet-18 FP32 | 46.016 | 49.335 | 33.34 |
| MobileNetV3-Small FP32 | 56.461 | 103.432 | 229.93 |

## Quantization benchmark

This is the quantization module’s repeated single-image measurement. It is kept
separate from the architecture benchmark because it creates and benchmarks each
quantized artifact in its own study:

| Variant | Artifact (MB) | AUC | Accuracy | Mean latency (ms/image) |
|---|---:|---:|---:|---:|
| FP32 | 17.667 | 0.9678 | 0.9183 | 104.502 |
| Dynamic INT8 | 16.688 | 0.9683 | 0.9199 | 105.910 |
| Static INT8 PTQ | 5.219 | 0.9427 | 0.7981 | 152.698 |

The manuscript must label these as two different measurement protocols rather than
mixing a model-comparison latency with a quantization-study latency.

## Memory benchmark

| Measurement | Recorded value |
|---|---:|
| Lazy streaming loader peak RSS | 134.4 MB |
| Naïve full-dataset loader peak RSS | 3,140.5 MB |
| Streaming reduction | 23.4× |
| FP32 isolated inference RSS delta | 544.4 MB |
| Dynamic INT8 isolated inference RSS delta | 563.2 MB |
| Static INT8 isolated inference RSS delta | 222.7 MB |

These are process-RSS observations, not universal hardware specifications. A future
rerun may differ slightly; if it does, create a new dated record and update every
manuscript table together.
