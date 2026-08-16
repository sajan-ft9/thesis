# Minimal Publication Plan

## Recommended paper identity

Use one focused paper question:

> **Can post-training INT8 quantization make an explainability-aware pediatric chest-X-ray pneumonia classifier substantially smaller and faster while preserving useful discrimination?**

Suggested title:

> **Memory-Efficient and Quantized Pneumonia Detection from Pediatric Chest X-rays: A Reproducible Evaluation with Visual Error Analysis**

This is narrower and more defensible than claiming clinical trust, universal generalization, or a new diagnostic model.

## Keep, reduce, and remove

### Keep as primary

- EfficientNet-B0 as the primary model.
- FP32 versus dynamic INT8 versus static INT8 PTQ.
- AUC, sensitivity, specificity, F1, calibration, size, latency, RSS, and the accuracy/efficiency trade-off.
- Leakage audit and reproducible configuration.
- One independent RSNA result as a clearly labelled exploratory appendix or short secondary analysis.

### Keep as secondary

- ResNet-18 and MobileNetV3-Small as compact comparator baselines.
- Grad-CAM++ examples for correct, false-positive, and false-negative cases.
- ONNX export as a software artifact, not proof of deployment.

### Remove or soften

- “Clinically usable,” “trustworthy,” “clinical accuracy,” and “human-in-the-loop oversight.” No reader study supports those claims.
- “Fully reproducible” unless raw-data checksums, environment lock files, and a clean rerun are documented.
- “Edge deployment” unless the manuscript says “workstation CPU proxy.”
- The claim that Grad-CAM++ explanations are clinically plausible. Call them model-derived visualizations or qualitative error-analysis maps.
- The claim that model selection was a priori unless it was genuinely registered before test results were viewed.

## Minimum experiments before submission

1. **Fix test-set threshold selection.** Select the operating threshold using validation predictions only, then lock it before evaluating the test set. Report test performance at the locked threshold. Keep the current test-derived Youden threshold only as an explicitly post-hoc descriptive analysis, or remove it.
2. **Repeat training with three seeds** for EfficientNet-B0 and the final quantization comparison. Report mean ± SD or seed-wise results. If compute is tight, repeat the primary model only and mark baseline results as single-seed exploratory.
3. **Measure actual artifact size.** Report `stat().st_size` for the exact `.pth`, `.pt`, or ONNX file used for deployment, alongside parameter count. Do not label an in-memory serialized state dict “on-disk model size.”
4. **Isolate RSS measurements.** Run each benchmark variant in a fresh subprocess, record absolute peak RSS and the measurement protocol, and distinguish process RSS from device memory.
5. **Make quantization provenance explicit.** Record backend, actual qconfig/observer, calibration sample count, calibration split, and whether the per-channel path or fallback path was used.
6. **Add a compact robustness table.** Evaluate the locked FP32 and INT8 thresholds on the same test set across three seeds; include AUC, sensitivity, specificity, F1, size, latency, and RSS.
7. **Complete CLAIM-style reporting.** Add a study-design statement, data source and patient/image split level, preprocessing, intended use, availability of code/model/data, funding/conflicts, and a full limitations paragraph. CLAIM 2024 specifically asks for these details; TRIPOD+AI is a useful second checklist for prediction-model reporting.

## Recommended paper structure

1. Introduction: resource-constrained CXR screening problem and narrow contribution.
2. Related work: Kermany, lightweight CNNs, PTQ, Grad-CAM++, and external generalization/confounding.
3. Methods: dataset, deduplication, split, model, quantization, calibration, locked threshold, metrics, benchmark protocol.
4. Results: primary FP32/INT8 table; repeated-seed variability; efficiency table; qualitative Grad-CAM error analysis; short RSNA appendix.
5. Discussion: what improved, what accuracy/calibration cost was paid, and why the result is not clinical validation.
6. Limitations and reproducibility: patient identity unavailable, single pediatric source, workstation proxy, no reader study, no prospective validation.

## Why these recommendations follow the literature

CLAIM 2024 recommends explicit study design, intended use, data source, split level, statistical analysis, and availability reporting. TRIPOD+AI similarly emphasizes transparent model development, validation, usability, and limitations. Zech et al. showed that pneumonia models can exploit hospital/source confounding and generalize poorly, so a high internal AUC cannot support broad clinical claims. The Grad-CAM++ paper supports visualization generation, but a visualization method alone does not establish faithfulness or clinical correctness. Quantization should be framed as an engineering trade-off consistent with integer-inference work such as Jacob et al.

References: [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300), [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), [Zech et al.](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683), [Grad-CAM++](https://arxiv.org/abs/1710.11063), [Jacob et al. quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html), and [Kermany et al.](https://pubmed.ncbi.nlm.nih.gov/29474911/).
