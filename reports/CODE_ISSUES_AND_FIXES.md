# Code Issues Affecting Publication

## Critical

### 1. Test-derived Youden threshold

`src/evaluate.py` computes `youden_threshold(y_true, y_prob)` from the held-out test predictions. The thesis presents this threshold as a deployment operating point. Selecting it from test labels makes the operating-point result optimistic and contaminates the test evaluation if treated as confirmatory.

**Fix:** generate validation predictions, select the threshold on validation only, save it in the checkpoint/config, then evaluate the locked threshold on test. Keep a separate `posthoc_test_youden` field only for descriptive analysis.

### 2. “On-disk model size” is not what the benchmark measures

`src/benchmarking.py:model_size_mb()` serializes `model.state_dict()` into `BytesIO`. The quantization rows therefore do not measure the actual `.pt` or `.pth` deployment artifact. `file_size_mb()` exists but is not used for the main result.

**Fix:** save each exact deployment artifact first and report its filesystem size. Report state-dict size separately if useful.

### 3. RSS is not isolated between variants

`benchmark_peak_rss()` samples the current process and returns peak-minus-baseline. `src/memory_profile.py` measures FP32, dynamic INT8, and static INT8 sequentially in that same process. Allocator caches, imported libraries, thread pools, and previous models can make deltas non-comparable.

**Fix:** benchmark each variant in a fresh subprocess, capture absolute peak RSS, record OS/device/thread settings, and run at least three repetitions. Label the result “process RSS,” not total device memory.

## Important

### 4. Static quantization fallback is not provenance-safe

`src/quantize.py:static_quantize()` silently falls back from the custom per-channel qconfig to the backend default qconfig after an exception. The output JSON records the backend but not the qconfig actually used.

**Fix:** return/record `qconfig_name`, observer classes, fallback exception, calibration batches, and backend. Fail the publication run if the required qconfig was not used.

### 5. Reproducibility is seed-controlled, not multi-seed

The pipeline uses seed 42 and deterministic settings, but one seed cannot estimate training variance. MPS/CUDA and library versions may still produce differences.

**Fix:** add a `--seed`/seed-list experiment runner and report at least three seeds for the primary model. Preserve each seed’s checkpoint and metrics.

### 6. Reporting generator wording (resolved)

The generator previously wrote “No external validation,” which conflicted with the completed RSNA probe. `src/reporting.py` now describes the result as limited exploratory external validation; it remains non-clinical and non-multi-centre.

### 7. Patient independence is not verified for Kermany

MD5 detects byte-identical files, not the same patient represented by different images or near-duplicates. The current split is image-level after byte deduplication.

**Fix:** if patient metadata cannot be recovered, state clearly that patient-level independence is unknown. Do not call the split patient-level.

## Moderate

### 8. Calibration and threshold claims are descriptive unless externally validated

Calibration slope, ECE, and Brier score are estimated on the test predictions. They are useful summaries but not a validated deployment calibration study. The RSNA Youden threshold is also post-hoc.

**Fix:** use validation for calibration/threshold selection and test only for final locked evaluation; describe test-fitted quantities as exploratory.

### 9. Multiple pairwise tests are uncorrected

Three pairwise DeLong and McNemar tests are reported at alpha 0.05. The ResNet-vs-MobileNet DeLong p=0.010 should be labelled uncorrected, or adjust with Holm and report both.

### 10. Grad-CAM++ is qualitative only

The implementation generates useful overlays, but no deletion/insertion, pointing-game, segmentation overlap, or reader agreement metric is computed.

**Fix:** either change the paper wording to “qualitative visual error analysis” or add a simple occlusion/deletion faithfulness analysis without claiming clinical correctness.

## Verification status

- `python3 -m compileall -q src scripts tests` passes.
- `python3 scripts/verify_thesis_numbers.py` passes the committed headline checks.
- The full pytest suite was not runnable in the current environment because `pytest` is not installed; install the project environment and run `make test` before submission.
