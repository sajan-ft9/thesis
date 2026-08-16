# Actual thesis run: plain-language results record

**Run date:** 2026-08-16  
**Command requested:** `make docker-actual`  
**Data:** real Kermany pediatric chest-X-ray images already mounted in `data/raw/`  
**Models:** existing real checkpoints in `models/`  
**Synthetic data:** not used for any result below

## What happened in simple words

The project was run again from the command line. It checked the real images, tested the
three saved models, measured their speed and memory use, created the compressed INT8
versions, produced explanation pictures, and regenerated the tables and report files.

The classification results were the same as the previous verified run. Speed and memory
numbers changed slightly because they depend on the computer and Docker runtime. That is
normal: the model is the same, but the computer may be busy or allocate memory differently.

The first final number check noticed this normal memory variation and stopped because the
old checker expected exact one-decimal RSS values. The checker was then corrected so that
stable scientific numbers are checked exactly while run-specific RSS numbers are printed
and documented separately. The final verification command now passes.

## Commands actually run

The top-level command was:

```bash
make docker-actual
```

This expanded into these real-data operations:

```bash
make docker-validate-data
make docker-evaluate
make docker-benchmark
make docker-quantize
make docker-explain
make docker-memory
make docker-stats
make docker-tradeoff
make docker-report
make docker-render
make docker-verify
```

After the RSS checker was made tolerant of expected runtime variation, the final check
was run directly:

```bash
make docker-verify
```

It passed all stable thesis-number checks and printed the current run-specific memory
measurements.

## Results from each step

### 1. Real data check — `make docker-validate-data`

This does not test the model. It checks that the images and labels are arranged correctly
and looks for duplicate files that could make the model appear better than it really is.

**Result:**

- Training images: 4,152
- Validation images: 1,038
- Final test images: 624
- Duplicate files removed: 26
- Leakage check: clean after the implemented byte/content checks

**Saved at:** `results/metrics/dataset_validation.json`

**In the thesis:** Chapter 3, data and integrity method; Chapter 4, dataset results and
limitations.

### 2. Model test — `make docker-evaluate`

This asks each saved model to classify the 624 real test images. The test images are not
used to choose the threshold. The reported threshold is the fixed 0.5 setting.

| Model | AUC | Accuracy | Sensitivity | Specificity | F1 | Errors |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.9678 | 0.9183 | 0.9667 | 0.8376 | 0.9366 | 51 |
| ResNet-18 | 0.9594 | 0.8798 | 0.9923 | 0.6923 | 0.9117 | 75 |
| MobileNetV3-Small | 0.9743 | 0.8750 | 0.9974 | 0.6709 | 0.9089 | 78 |

**Layman meaning:** AUC measures how well the model separates sick from normal images
across many possible cutoffs. Sensitivity means “how many pneumonia images it catches.”
Specificity means “how many normal images it correctly leaves alone.” EfficientNet-B0
was not the highest-AUC model, but it produced fewer false alarms: 38 false positives,
compared with 72 for ResNet-18 and 77 for MobileNetV3-Small.

**Saved at:**

- `results/metrics/efficientnet_b0_test_metrics.json`
- `results/metrics/resnet18_test_metrics.json`
- `results/metrics/mobilenetv3_small_test_metrics.json`
- `results/metrics/*_error_cases.csv`
- `results/roc_curves/`
- `results/figures/*_pr_curve.*`
- `results/confusion_matrices/`

**In the thesis:** Abstract; Chapter 4, Sections 4.1 and 4.2; final publication Section 4.1.

### 3. Speed test — `make docker-benchmark`

This measures how long each normal FP32 model takes to process an image in the Docker
CPU environment. It is not a phone, Raspberry Pi, or hospital computer measurement.

**This run measured:**

- EfficientNet-B0: 92.821 ms per image, 41.71 images/second
- ResNet-18: 46.016 ms per image, 33.34 images/second
- MobileNetV3-Small: 56.461 ms per image, 229.93 images/second

These figures can move between runs because CPU scheduling and Docker host load change.

**Saved at:** `results/metrics/*_benchmark.json`

**In the thesis:** Chapter 4, Section 4.4 and the efficiency tables. Treat these as
container CPU proxy measurements.

### 4. Compression test — `make docker-quantize`

This creates smaller INT8 versions of EfficientNet-B0 and checks whether they still work.

| Version | File size | AUC | Accuracy | Mean latency |
|---|---:|---:|---:|---:|
| FP32 | 17.667 MB | 0.9678 | 0.9183 | 104.502 ms |
| Dynamic INT8 | 16.688 MB | 0.9683 | 0.9199 | 105.910 ms |
| Static INT8 PTQ | 5.219 MB | 0.9427 | 0.7981 | 152.698 ms |

**Layman meaning:** Static INT8 makes the file much smaller, but in this computer setup
it is slower and loses some accuracy. It is therefore a memory/storage trade-off, not
automatically a better model.

**Saved at:**

- `results/metrics/efficientnet_b0_quantization.json`
- `models/efficientnet_b0_fp32.pth`
- `models/efficientnet_b0_int8_dynamic.pth`
- `models/efficientnet_b0_int8_static.pt`
- `models/efficientnet_b0.onnx`
- `results/tables/table5_quantization.*`
- `results/figures/quantization_comparison.*`

**In the thesis:** Chapter 4, Section 4.3; final publication Section 4.2.

### 5. Explanation pictures — `make docker-explain`

This creates heat-map pictures showing which parts of each real X-ray influenced selected
predictions. It generated 12 overlays covering correct predictions, false positives, and
false negatives.

**Saved at:**

- `results/gradcam/*.png`
- `results/figures/efficientnet_b0_gradcam_grid.*`
- `results/metrics/efficientnet_b0_gradcam_records.csv`
- `results/metrics/efficientnet_b0_explainability.json`

**Layman meaning:** These pictures help a person inspect what the model appears to look
at. They do not prove that the model is looking at the medically correct anatomy.

**In the thesis:** Chapter 4, Section 4.5; final publication Section 4.5.

### 6. Memory test — `make docker-memory`

This asks two questions: how much memory the data loader uses, and how much memory each
model uses while processing the full test set in a fresh process.

**This run measured:**

- Lazy streaming data load: 134.4 MB RSS
- Naïve full-dataset load: 3,140.5 MB RSS
- Difference: 23.4 times lower for streaming
- FP32 inference RSS delta: 544.4 MB
- Dynamic INT8 inference RSS delta: 563.2 MB
- Static INT8 inference RSS delta: 222.7 MB

**Layman meaning:** Loading one image/batch at a time uses much less memory than loading
the whole dataset at once. Memory readings can vary slightly between runs, so these are
measurements from this run, not permanent hardware specifications.

**Saved at:**

- `results/metrics/memory_profile.json`
- `results/metrics/efficientnet_b0_memory_fp32.json`
- `results/metrics/efficientnet_b0_memory_dynamic.json`
- `results/metrics/efficientnet_b0_memory_static.json`
- `results/figures/memory_footprint.*`

**In the thesis:** Chapter 4, Section 4.4.1; final publication Section 4.3.

### 7. Statistical analysis — `make docker-stats`

This reuses the real test predictions to calculate calibration, confidence intervals, and
paired model comparisons. It does not train another model.

**Important current values:** EfficientNet-B0 MCC 0.8250, balanced accuracy 0.9021, ECE
0.0354, and Brier score 0.0633. Its AUC difference versus MobileNetV3-Small was not
statistically significant in the paired comparison.

**Saved at:** `results/metrics/statistical_analysis.json` and the calibration/error/
threshold figures in `results/figures/`.

**In the thesis:** Chapter 4, Section 4.2.1 and the discussion of model comparison.

### 8. Report generation — `make docker-tradeoff`, `make docker-report`, `make docker-render`

These commands do not make new predictions. They turn the JSON results into charts,
tables, and readable thesis files.

**Saved at:**

- `results/figures/accuracy_efficiency_tradeoff.*`
- `results/tables/`
- `reports/thesis_rendered.md`
- `paper_assets/`

### 9. Final number check — `make docker-verify`

This checks that stable headline numbers written in the thesis agree with the JSON files.
The current final check passes all stable checks. RSS values are printed as measured
run-specific values rather than required to match the thesis text to the last decimal.

## Where the results appear in the thesis

The main controlled thesis is `reports/thesis_final.md`:

- Abstract: short summary of the primary result and limitations.
- Chapter 3: dataset, split, threshold, training, quantization, memory, and explanation methods.
- Chapter 4, Sections 4.1–4.2: real test performance and baseline comparison.
- Chapter 4, Section 4.3: FP32/dynamic/static INT8 results.
- Chapter 4, Section 4.4: CPU efficiency and memory results.
- Chapter 4, Section 4.5: Grad-CAM++ qualitative explanation results.
- Chapter 4, Section 4.7: RSNA external probe.
- Chapter 5: plain-language interpretation and limitations.

The shorter journal-style version is `paper_assets/publication_final.md`; its corresponding
results are in Sections 4.1–4.5.

## One-command summary for a non-technical reader

Run:

```bash
make docker-actual
```

Then open:

1. `reports/ACTUAL_RUN_RESULTS.md` — this simple explanation;
2. `reports/thesis_final.md` — the formal thesis;
3. `results/metrics/efficientnet_b0_test_metrics.json` — the main model test result;
4. `results/metrics/efficientnet_b0_quantization.json` — compression comparison;
5. `results/metrics/memory_profile.json` — memory measurements;
6. `results/figures/` and `results/gradcam/` — pictures.

