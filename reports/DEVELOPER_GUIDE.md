# Developer Guide

## What this project is

This repository is a reproducible research pipeline for binary pneumonia classification from chest X-rays. It trains ImageNet-initialised CNNs, evaluates them on the Kermany pediatric dataset, runs an exploratory zero-shot RSNA experiment, generates Grad-CAM++ images, compares FP32/dynamic-INT8/static-INT8 variants, and measures CPU latency, model size, and process RSS.

The main model is EfficientNet-B0. ResNet-18 and MobileNetV3-Small are comparators. The project is a research artifact, not a clinical device.

## Where things live

| Area | Location | Purpose |
|---|---|---|
| Configuration | `configs/*.yaml` | Seeds, transforms, model, training, evaluation and benchmark settings |
| Core code | `src/` | Dataset integrity, training, inference, metrics, quantization, explainability and reporting |
| Entry points | `Makefile`, `scripts/` | Smoke test, validation, reproduction and external-data preparation |
| Tests | `tests/` | Offline unit and synthetic integration checks |
| Metrics | `results/metrics/` | JSON provenance for reported numbers |
| Figures/tables | `results/figures/`, `results/tables/` | Generated outputs |
| Thesis | `reports/thesis_final.md` | Rendered thesis with committed numbers |

## How data flows

1. `src.dataset` scans `train/{NORMAL,PNEUMONIA}` and `test/{NORMAL,PNEUMONIA}`.
2. Byte-identical files are hashed and duplicate training files are removed before the stratified 80/20 train/validation split.
3. Training uses augmentation; validation/test use deterministic resize and ImageNet normalization.
4. `src.train` fine-tunes the configured backbone and selects the checkpoint by validation AUC.
5. `src.evaluate` applies the untouched Kermany test set and writes metrics, CIs, ROC/PR curves, confusion matrices, and error cases.
6. `src.quantize` calibrates static PTQ on clean validation data; it does not use the test set.
7. `src.benchmarking` measures serialized state-dict size, warm-up-corrected latency, throughput, and sampled RSS.
8. `scripts/build_rsna_subset.py` converts RSNA DICOM files to PNG and maps `Normal` to NORMAL and `Lung Opacity` to PNEUMONIA; `No Lung Opacity / Not Normal` is excluded.

## Reproduction commands

```bash
make setup
make smoke                 # synthetic data only; never report its metrics
make validate-data
make reproduce             # real data; training is compute-intensive
make verify-numbers
```

The committed primary results report Kermany test AUC 0.9678 (95% CI 0.9504–0.9816), static INT8 size 5.22 MB, static PTQ AUC 0.9427, and RSNA exploratory AUC 0.8892 (95% CI 0.8825–0.8954). These are artifact values, not guarantees that a new run will be bit-identical across hardware or library versions.

## Developer cautions

- `make verify-numbers` checks text-to-JSON consistency; it cannot prove raw-data provenance or prevent selective reporting.
- The current manifests record a Git SHA, seed, environment and configuration, but the raw datasets are not committed. Preserve dataset download source, checksums, and preprocessing logs for a publishable rerun.
- Byte hashes do not detect near-duplicate or same-patient images. Patient identifiers and perceptual-near-duplicate checks are still missing.
- The RSNA mapping is useful for an exploratory binary probe, not a clinical definition of pneumonia.
- RSS is a process-level proxy and depends on the host OS, allocator, thread count, and background processes. Do not present it as device RAM or power consumption.
- Grad-CAM++ is qualitative. It is not evidence that the model uses clinically correct anatomy.
- Quantization is backend-dependent. Always report backend, calibration data, accuracy/AUC cost, and threshold-specific operating-point changes together.

## Minimum checks before changing the project

Run `make smoke`, `make verify-numbers`, and the relevant tests. If changing data, transforms, thresholds, quantization, or model selection, regenerate affected JSON, figures, tables, and the rendered thesis; do not hand-edit headline numbers.
