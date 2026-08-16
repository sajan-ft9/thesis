# Project Runbook: actual research, software tests, commands, and outputs

This document is the operational map for the repository. It distinguishes outputs that
can be reported as research results from outputs that exist only to test the software.
The Docker-backed `make docker-*` targets are the recommended commands because they use
the pinned CPU environment and mount the local data, model, result, report, and paper
directories.

## 1. The important distinction: actual versus test

| Category | Make targets | Input | Can its numbers appear in the thesis? |
|---|---|---|---|
| Software tests | `make docker-test` | Unit tests and synthetic fixtures | **No.** These prove code behavior only. |
| Tiny smoke test | `make docker-smoke` | Generated tiny synthetic dataset | **No.** It proves the pipeline can execute end-to-end. |
| Primary research | `make docker-actual` | Real Kermany images and existing real checkpoints | **Yes**, subject to the stated limitations. |
| Full real reproduction | `make docker-reproduce` | Real Kermany images; retrains all models | **Yes**, after inspecting the regenerated metrics and manifests. |
| External robustness probe | `make docker-external-rsna` | Real RSNA-derived processed images | **Yes, but only as exploratory external evidence**, not clinical validation. |
| Single-image demo | `make docker-inference` | One user-provided image at `input/image.png` | **No.** It is an inference demonstration, not an evaluation. |

The words “real” and “actual” refer to the dataset and checkpoint inputs. They do not
mean that the result is clinically validated. The primary result remains an in-dataset
held-out test result; the RSNA result is a domain-shift probe.

## 2. First-time setup

### 2.1 Obtain the real data

Place the Kermany dataset at `data/raw/chest_xray/` with the expected `train/` and
`test/` class directories. See `data/README.md` for the source and licensing notes.
The repository intentionally does not embed medical images.

Optional RSNA data is built separately by `make docker-external-rsna`; it can require a
large download and network access.

### 2.2 Build the environment

```bash
make docker-build
```

This builds the CPU image `pneumonia-edge-xai:cpu`. The Compose file uses a Linux x86
platform and a 2 GB shared-memory mount so PyTorch DataLoader workers work reliably on
macOS/ARM hosts.

## 3. Recommended command sequences

### 3.1 Software-only confidence check

```bash
make docker-test
make docker-smoke
```

`docker-test` runs the unit and integration suite. `docker-smoke` runs a tiny synthetic
end-to-end pipeline. Both are expected to create or overwrite small generated files,
but neither result may be quoted as evidence for model performance.

### 3.2 Actual analysis using the existing real checkpoints

```bash
make docker-actual
```

This is the normal command when the trained checkpoints already exist in `models/`.
It validates the real data, evaluates the three real checkpoints, benchmarks them,
quantizes EfficientNet-B0, measures memory, generates statistics and explanations,
regenerates tables/reports, and verifies thesis numbers. It does **not** retrain models.

### 3.3 Full actual reproduction including training

```bash
make docker-reproduce
```

This first trains EfficientNet-B0, ResNet-18, and MobileNetV3-Small on the real Kermany
data, then runs the complete actual analysis. It can take substantially longer than
`docker-actual` and overwrites the corresponding checkpoint files and generated metrics.
Review the new metadata and metrics before treating the run as the publication run.

### 3.4 Optional RSNA external probe

```bash
make docker-external-rsna
make docker-report
make docker-render
make docker-verify
```

This fetches/prepares the real RSNA-derived probe and evaluates EfficientNet-B0 without
threshold tuning. It is intentionally separate because it may require a large download
and its label mapping is only an exploratory binary proxy.

### 3.5 Single-image inference

Put an image at `input/image.png`, then run:

```bash
make docker-inference
```

This loads `models/efficientnet_b0_best.pth`, prints the predicted class and probability,
and demonstrates deployment wiring. It is not a test-set result and must not be used to
calculate publication metrics.

## 4. Command-by-command output map

### `make docker-validate-data`

**What it does:** scans the real Kermany data, applies the project’s duplicate policy,
checks split membership and content overlap, and records class counts.

**How it works:** `src.dataset` hashes the available images and constructs the seeded
train/validation/test view. The official test directory remains held out.

**Primary output:**

- `results/metrics/dataset_validation.json`

**What to inspect:** `split_sizes`, `class_distribution`, `n_train_duplicates_removed`,
`raw_split_sizes`, and the leakage/duplicate records.

**Verified project result:** 4,152 train / 1,038 validation / 624 test images, with 26
byte-identical duplicate files removed. This is a data-integrity result, not a model
performance result.

### `make docker-train`

**What it does:** trains the primary EfficientNet-B0 checkpoint on the real training
partition and selects the best epoch using validation AUC.

**Outputs:**

- `models/efficientnet_b0_best.pth`
- `results/metrics/efficientnet_b0_history.json`
- `results/metrics/efficientnet_b0_history.csv`
- `results/metrics/efficientnet_b0_metadata.json`
- `results/tensorboard/efficientnet_b0/`

**What to inspect:** the metadata seed/configuration, best epoch, training/validation
curves, and whether the output checkpoint timestamp matches the run being reported.
The history is training evidence; it is not the held-out test result.

### `make docker-train-baselines`

**What it does:** trains the real ResNet-18 and MobileNetV3-Small comparison models using
their committed configurations.

**Outputs:** corresponding `models/*_best.pth`, history files, metadata files, and
TensorBoard directories for both models.

**What to inspect:** identical split/configuration policy, best validation epoch, and
checkpoint provenance. Baseline training results are only comparable if the same data
and evaluation policy are used.

### `make docker-evaluate`

**What it does:** evaluates existing checkpoints once on the real held-out Kermany test
set. The decision threshold is the locked configuration value 0.5. Validation predictions
are used only for a separate exploratory Youden diagnostic; the test set is not used to
select the reported threshold.

**Outputs per model:**

- `results/metrics/<model>_test_metrics.json`
- `results/metrics/<model>_error_cases.csv`
- `results/roc_curves/<model>_roc_curve.{png,pdf}`
- `results/figures/<model>_pr_curve.{png,pdf}`
- `results/confusion_matrices/<model>_confusion_matrix.{png,pdf}`

**What to inspect in JSON:** `metrics`, `confidence_intervals`, `threshold`,
`threshold_source`, `validation_youden_threshold`, and the false-positive/false-negative
counts. The JSON is the source of truth for model performance; tables and prose are
derived from it.

**Verified EfficientNet-B0 result:** AUC 0.9678 (95% CI 0.9504–0.9816), accuracy 0.9183,
sensitivity 0.9667, specificity 0.8376, and F1 0.9366.

### `make docker-benchmark`

**What it does:** measures real checkpoint artifact size, warm-up-corrected repeated
CPU latency, latency variation, throughput, and sampled process RSS.

**Outputs:**

- `results/metrics/efficientnet_b0_benchmark.json`
- `results/metrics/resnet18_benchmark.json`
- `results/metrics/mobilenetv3_small_benchmark.json`
- generated efficiency figures/tables through `docker-report`

**What to inspect:** `size_mb`, `latency_ms_mean`, `latency_ms_std`,
`latency_ms_p95`, `throughput_img_per_s`, and `peak_rss_delta_mb`.

**Publication benchmark record:** EfficientNet-B0 92.821 ms/image and 41.71 images/s;
ResNet-18 46.016 ms/image; MobileNetV3-Small 56.461 ms/image. These are container CPU
proxy measurements, not physical Raspberry Pi/Jetson results.

### `make docker-quantize`

**What it does:** creates and evaluates FP32, dynamic INT8, and static INT8 PTQ versions
of the real EfficientNet-B0 checkpoint. Static PTQ uses clean validation calibration,
per-channel symmetric weights, histogram activation observers, and QNNPACK.

**Outputs:**

- `models/efficientnet_b0_fp32.pth`
- `models/efficientnet_b0_int8_dynamic.pth`
- `models/efficientnet_b0_int8_static.pt`
- `models/efficientnet_b0.onnx`
- `results/metrics/efficientnet_b0_quantization.json`
- `results/figures/quantization_comparison.{png,pdf}`
- `results/tables/table5_quantization.{csv,md,tex}`

**What to inspect:** each `variants` row, especially `artifact_size_mb`, `accuracy`,
`auc`, `latency_ms_mean`, `size_reduction_pct`, `auc_drop_pct`, and `quantization`.
The static result is not automatically “better”: it trades size/RSS for accuracy and
latency in this backend.

**Publication benchmark record:** FP32 17.667 MB / AUC 0.9678 / 104.502 ms; dynamic INT8 16.688 MB /
AUC 0.9683 / 105.910 ms; static INT8 5.219 MB / AUC 0.9427 / 152.698 ms.

### `make docker-memory`

**What it does:** measures two separate real-data memory questions. First, it compares
lazy streaming against naïve full-dataset float32 loading. Second, it launches a fresh
subprocess for each precision variant and samples process RSS during full test inference.

**Outputs:**

- `results/metrics/memory_profile.json`
- `results/metrics/efficientnet_b0_memory_fp32.json`
- `results/metrics/efficientnet_b0_memory_dynamic.json`
- `results/metrics/efficientnet_b0_memory_static.json`
- `results/figures/memory_footprint.{png,pdf}`

**What to inspect:** `streaming_vs_naive`, `runtime_memory.variants`, `measurement`,
`artifact_size_mb`, `inference_peak_rss_mb`, and `quantization` provenance.

**Publication benchmark record:** streaming 134.4 MB versus naïve 3,140.5 MB, a 23.4-fold measured
difference. Fresh-process inference RSS deltas were FP32 544.4 MB, dynamic INT8 563.2 MB,
and static INT8 222.7 MB. RSS is a process-level proxy and depends on the platform.

### `make docker-stats`

**What it does:** calculates extended metrics, calibration, paired bootstrap AUC
comparisons, DeLong tests, McNemar tests, threshold robustness, and reliability curves
from real held-out predictions.

**Outputs:**

- `results/metrics/statistical_analysis.json`
- `results/figures/calibration_reliability.{png,pdf}`
- `results/figures/error_distribution.{png,pdf}`
- `results/figures/threshold_robustness.{png,pdf}`

**What to inspect:** `per_model_kermany`, `primary_full_cis`, and `pairwise_kermany`.
These are inference-only comparisons on one test set. Pairwise p-values are exploratory,
not a replacement for prospective clinical validation or a multiplicity-adjusted trial.

### `make docker-explain`

**What it does:** generates Grad-CAM++ overlays for correct, false-positive, and
false-negative real test cases.

**Outputs:**

- `results/gradcam/*.png`
- `results/figures/efficientnet_b0_gradcam_grid.{png,pdf}`
- `results/metrics/efficientnet_b0_gradcam_records.csv`
- `results/metrics/efficientnet_b0_explainability.json`

**What to inspect:** the case type, true label, predicted label, probability, and image
overlay. These are qualitative error-analysis outputs. They do not prove anatomical
faithfulness or clinical trustworthiness because no radiologist localization study was
performed.

### `make docker-tradeoff`

**What it does:** combines current real performance and efficiency metrics into a
deployment-frontier figure.

**Output:** `results/figures/accuracy_efficiency_tradeoff.{png,pdf}`.

This figure is a decision aid. It does not establish a universal best model because the
choice depends on the desired balance of AUC, false positives, sensitivity, storage,
latency, and memory.

### `make docker-report`

**What it does:** reads the current JSON metrics and regenerates Markdown, CSV, and LaTeX
tables plus report artifacts.

**Outputs:**

- `results/tables/table1_dataset_statistics.*`
- `results/tables/table2_training_config.*`
- `results/tables/table3_model_comparison.*`
- `results/tables/table4_final_performance.*`
- `results/tables/table5_quantization.*`
- `results/tables/table6_efficiency.*`
- generated report files under `reports/` and `paper_assets/`

This target should be run after any change to metrics, evaluation, quantization, or
benchmarking. Do not hand-edit generated headline numbers.

### `make docker-render`

**What it does:** regenerates `reports/thesis_rendered.md` from the current thesis source
and current result JSON files.

**Outputs:** `reports/thesis_rendered.md`.

The rendered thesis is a convenience copy. `reports/thesis_final.md` is the controlled
manuscript checked by the number verifier.

### `make docker-verify`

**What it does:** checks that the headline numbers written in `reports/thesis_final.md`
occur exactly in the current result JSON artifacts.

**Output:** terminal PASS/FAIL report; current verified run is 31/31 checks passing.

This verifies internal text-to-artifact consistency. It cannot prove dataset provenance,
patient independence, clinical validity, or absence of selective reporting.

## 5. Complete output directory map

| Path | Meaning |
|---|---|
| `data/raw/` | Real source datasets; not committed. |
| `data/processed/` | Derived real external data, especially RSNA probe files. |
| `models/` | Checkpoints and quantized/exported artifacts. |
| `results/metrics/` | Source-of-truth JSON/CSV measurements and manifests. |
| `results/tables/` | Generated Markdown/CSV/LaTeX tables. |
| `results/figures/` | PR, calibration, error, memory, quantization, Grad-CAM, and trade-off figures. |
| `results/roc_curves/` | ROC figures. |
| `results/confusion_matrices/` | Confusion-matrix figures. |
| `results/gradcam/` | Individual Grad-CAM++ overlays. |
| `reports/` | Thesis, audit, limitations, developer, reader, and publication documentation. |
| `paper_assets/` | Submission manuscript and paper-support files. |

## 6. Current verified headline values

These values came from the committed real-data run and are not from the synthetic test
suite:

- EfficientNet-B0 Kermany test AUC: 0.9678 (95% CI 0.9504–0.9816).
- EfficientNet-B0 accuracy/sensitivity/specificity/F1: 0.9183 / 0.9667 / 0.8376 / 0.9366.
- Static INT8 artifact/AUC/latency: 5.219 MB / 0.9427 / 152.698 ms.
- Streaming versus naïve RSS: 134.4 MB versus 3,140.5 MB.
- RSNA exploratory AUC: 0.8892 (95% CI 0.8825–0.8954).
- Software tests: 40 passing; thesis number checks: 31/31 passing.

## 7. Safe reporting rules

Do not report smoke-test numbers. Do not call the RSNA probe clinical validation. Do not
call RSS device memory or latency universal. Do not claim static INT8 is faster based only
on its smaller file size. Always report the dataset, split, checkpoint, threshold,
backend, calibration data, and output JSON path alongside a number.
