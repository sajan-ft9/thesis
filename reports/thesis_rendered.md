<!--
  RESULTS-DRIVEN THESIS TEMPLATE
  ------------------------------------------------------------------------------
  This document contains NO hard-coded experimental numbers. Every result is a
  {{<token>}}-style placeholder that is filled from results/metrics/*.json by:

      python scripts/render_report.py --config configs/efficientnet_b0.yaml
      -> writes reports/thesis_rendered.md

  Until you run the pipeline on the real Kermany dataset, placeholders remain
  visible (and render_report.py lists them as "pending"). This guarantees the
  thesis can never silently contain fabricated or stale numbers.

  Tables 1-6 are generated in results/tables/ as .md / .csv / .tex — paste the
  rendered versions in, or \input the .tex files in a LaTeX build.
-->

# Memory-Efficient Explainable Pneumonia Detection Using Quantized EfficientNet-B0 for Resource-Constrained Healthcare Environments

**Author:** Rohan Maleku
**Degree:** Master of Science in Information Technology
**Institution:** [Your University]
**Supervisor:** [Supervisor Name]
**Date:** [Submission Date]

> ⚠️ **RESULTS STATUS:** Placeholders in this document (e.g. `0.9678`) are
> populated only after running `make reproduce` on the Kermany dataset. Do not
> submit until `scripts/render_report.py` reports **0 pending** placeholders.

---

## Abstract

**Background.** Pneumonia is a leading cause of childhood mortality, with the
greatest burden in low- and middle-income countries where access to expert chest
X-ray (CXR) interpretation is limited. Deep learning can assist screening, but
high-capacity models are computationally heavy and opaque, limiting deployment on
constrained hardware and eroding clinical trust.

**Objective.** This thesis designs, implements, and *reproducibly* evaluates a
framework that jointly addresses three axes — diagnostic performance,
computational efficiency, and explainability — for binary pneumonia detection on
CXR images, with an emphasis on resource-constrained deployment.

**Methods.** The framework uses an ImageNet-pretrained **EfficientNet-B0** backbone
with a custom binary head, trained with a staged transfer-learning schedule, mixed
precision, label smoothing, cosine learning-rate scheduling, and early stopping.
Identical pipelines train **ResNet-18** and **MobileNetV3-Small** baselines for a
fair architectural comparison. Evaluation uses the held-out Kermany test set with
**bootstrap 95% confidence intervals**. **Grad-CAM++** provides visual
explanations for correctly classified, false-positive, and false-negative cases.
Two post-training quantization schemes — **dynamic INT8** and **static INT8 (PTQ
with calibration)** — are compared on size, latency, and accuracy. Efficiency is
benchmarked correctly (on-disk model size; warmup-corrected, repeated latency
timing; throughput; sampled peak process memory) — explicitly avoiding
`tracemalloc`, which understates true memory use.

**Results.** On the held-out test set (n = 624), EfficientNet-B0 achieves
ROC-AUC = 0.9678 (95% CI [0.9504, 0.9816]), accuracy = 0.9183,
sensitivity = 0.9667, and specificity = 0.8376. Static INT8
quantization reduces model size from 17.67 MB to 5.22 MB
(70.5% reduction) with an accuracy change of
10.740% and an AUC change of 2.340%, whereas
dynamic INT8 reduces size only to 16.69 MB (Linear layers only).
Full per-model and per-variant results are in Tables 3–6.

**Conclusion.** Competitive pneumonia screening is achievable within tight
resource budgets when lightweight architecture, integrated explainability, and
appropriate quantization are combined and measured rigorously. The complete code,
configs, and reproducibility artifacts are released to support replication and
extension.

**Keywords:** pneumonia detection, chest X-ray, EfficientNet, Grad-CAM++,
post-training quantization, edge AI, explainable AI, reproducible research.

---

## Table of Contents

1. [Introduction](#chapter-1-introduction)
2. [Literature Review](#chapter-2-literature-review)
3. [Methodology](#chapter-3-methodology)
4. [Results and Analysis](#chapter-4-results-and-analysis)
5. [Discussion](#chapter-5-discussion)
6. [Conclusion and Future Work](#chapter-6-conclusion-and-future-work)
7. [References](#references)
8. [Appendices](#appendices)

---

## Chapter 1: Introduction

### 1.1 Background and Motivation

Pneumonia accounts for a substantial share of deaths in children under five, with
the highest burden in low-resource settings where radiological expertise is scarce
[1, 2]. CXR is the most accessible imaging modality for pneumonia, but reliable
interpretation requires expertise that is frequently unavailable at the point of
care [3]. Automated CXR analysis with deep learning is therefore attractive for
screening — provided the models are *deployable* on the modest hardware available
in such settings and *transparent* enough to support clinician oversight.

Three coupled barriers motivate this work: (1) the **computational burden** of
large models; (2) **memory bottlenecks** in naïve data pipelines; and (3) the
**opacity** of deep models, which limits clinical trust [5–7].

### 1.2 Research Objectives

1. **Diagnostic performance.** Train and rigorously evaluate a lightweight CXR
   pneumonia classifier, reporting metrics with confidence intervals rather than
   point estimates alone. (Target reference: ROC-AUC ≥ 0.95; treated as an
   objective, not an assumed outcome.)
2. **Computational efficiency.** Quantify model size, inference latency, and
   throughput, and reduce them via post-training quantization suitable for edge
   deployment.
3. **Explainability.** Integrate Grad-CAM++ to visualise the image regions driving
   each prediction, including failure cases.
4. **Reproducibility.** Provide a fully scripted, seed-controlled, configuration-
   driven pipeline whose every reported number is regenerable from raw data.

### 1.3 Contributions

1. An **integrated, reproducible framework** combining EfficientNet-B0,
   Grad-CAM++, and INT8 quantization, with a fair baseline comparison against
   ResNet-18 and MobileNetV3-Small under an identical pipeline.
2. A **rigorous efficiency methodology** — correct latency/throughput timing and
   memory measurement — that corrects common benchmarking errors.
3. A **transparent quantization study** comparing dynamic vs static PTQ, reporting
   accuracy and AUC before and after quantization.
4. An **open, defensible artifact set**: configs, tests, metadata manifests, and
   auto-generated tables/figures, with no simulated or fabricated results.

### 1.4 Thesis Structure

Chapter 2 reviews related work; Chapter 3 details methodology; Chapter 4 reports
results; Chapter 5 discusses implications and limitations; Chapter 6 concludes and
outlines future work.

---

## Chapter 2: Literature Review

### 2.1 Deep Learning for Chest X-Ray Analysis

CNNs with ImageNet pretraining have driven progress in CXR analysis [8].
Rajpurkar et al. introduced CheXNet, a 121-layer DenseNet reporting
radiologist-level pneumonia detection on ChestX-ray14 [4]. Such models prioritise
accuracy and are parameter- and memory-heavy, complicating low-resource
deployment [5].

### 2.2 Lightweight Architectures

Depthwise-separable convolutions and compound scaling (MobileNetV2/V3 [12];
EfficientNet [13]) reduce parameters while retaining accuracy, making them
attractive backbones for constrained medical applications [14].

### 2.3 Explainable AI in Clinical Decision Support

CAM [15], Grad-CAM [16], and Grad-CAM++ [17] produce saliency maps that localise
the evidence behind a prediction. In medical imaging, such maps support error
analysis and clinician oversight, but require human validation before any claim of
clinical alignment [18, 19]. **No such human validation is claimed in this thesis**
(see Limitations).

### 2.4 Quantization and Edge Deployment

Post-training quantization (FP32 → INT8) reduces size and can accelerate CPU
inference [20]. *Dynamic* quantization typically targets Linear layers only, while
*static* PTQ with calibration also quantizes convolutions — important for
convolution-heavy backbones. ONNX Runtime and TensorFlow Lite enable cross-
platform inference on ARM edge devices [21].

### 2.5 Research Gap

Individual ingredients are well studied, but integrated, *correctly benchmarked*,
and *reproducible* pipelines that jointly report performance, efficiency, and
explainability for constrained CXR deployment remain uncommon. This thesis targets
that gap with an explicit emphasis on measurement rigor and reproducibility.

---

## Chapter 3: Methodology

### 3.1 Dataset and Splits

We use the **Kermany pediatric CXR dataset** [22] (classes: Normal, Pneumonia).
Dataset statistics are generated automatically (Table 1). The official dataset
ships a very small validation split (16 images); rather than rely on it for model
selection, we carve a **stratified validation split** from the training set
(fraction `data.val_split`) and keep the **official test set untouched** for final
evaluation. Before training, an automated check reports per-split class balance,
**byte-identical duplicate images**, and **train/val/test leakage** (by path and
content hash); the report is saved to `results/metrics/dataset_validation.json`.

- Total images: 5840 (Normal 1575, Pneumonia 4265).
- Split sizes: train 4172, validation 1044, test 624.

*Table 1 — Dataset Statistics: `results/tables/table1_dataset_statistics.md`.*

### 3.2 Preprocessing and Augmentation

Images are resized and normalised with ImageNet statistics. **Training**
augmentation (configurable) comprises resize-then-random-crop to 224×224, random
horizontal flip, small rotation, and brightness/contrast jitter.
**Validation/test** use deterministic resize + normalise only (no augmentation),
ensuring unbiased, reproducible evaluation. All transforms are defined in
`configs/` (Section 3.7).

### 3.3 Model Architecture

The backbone is **EfficientNet-B0** (ImageNet-pretrained), with its classifier
replaced by a shared binary head:

```
BatchNorm1d(F) → Dropout(0.3) → Linear(F, 256) → ReLU → Dropout(0.2) → Linear(256, 1)
```

where F is the backbone feature dimension. A single logit is trained with binary
cross-entropy; sigmoid at inference yields a probability. The identical head and
training pipeline are applied to **ResNet-18** and **MobileNetV3-Small** for a
controlled comparison (Table 3).

### 3.4 Transfer-Learning Schedule

Early epochs freeze the leading backbone stages and train the head; from
`train.unfreeze_epoch` the full backbone is unfrozen and fine-tuned at a reduced
learning rate. This stabilises early training and adapts features to CXR data.

### 3.5 Training Protocol

- Optimiser: AdamW; weight decay `train.weight_decay`.
- Schedule: linear warmup then cosine annealing to `train.lr_min`.
- Loss: BCE with label smoothing (`train.label_smoothing`).
- Mixed precision (CUDA), gradient clipping, early stopping on validation AUC.
- Full configuration is recorded per run (Table 2) and in a `metadata.json`
  reproducibility manifest (timestamp, git SHA, seed, library versions, config).

*Table 2 — Training Configuration: `results/tables/table2_training_config.md`.*

### 3.6 Evaluation and Statistical Reporting

On the held-out test set we report accuracy, precision, recall/sensitivity,
specificity, F1, and ROC-AUC, plus **non-parametric bootstrap 95% confidence
intervals** for accuracy and ROC-AUC (`evaluate.bootstrap_n` resamples). We also
compute the **Youden-J optimal threshold** and a per-error (FP/FN) table for
qualitative analysis. Confusion-matrix, ROC, and Precision-Recall figures are
exported as PNG and PDF.

### 3.7 Explainability

**Grad-CAM++** overlays are generated for the predicted class on three categories
sampled from real test predictions: correctly classified, false-positive, and
false-negative cases. Overlays and metadata (true/predicted label, probability)
are saved for auditability. No radiologist ratings, trust scores, or clinical
validations are produced or simulated.

### 3.8 Quantization for Edge Deployment

Two post-training schemes are compared (Table 5):
1. **Dynamic INT8** — quantizes Linear-layer weights.
2. **Static INT8 (PTQ)** — FX graph-mode quantization with a calibration pass,
   quantizing convolutions as well (backend: qnnpack).

For each, model size and test accuracy/AUC are reported before vs after. The FP32
model is also exported to **ONNX** (verified via ONNX Runtime). Where static PTQ
is unsupported on the host backend, this is reported rather than estimated.

### 3.9 Efficiency Benchmarking

Efficiency is measured correctly (Table 6): **on-disk model size**; **latency** at
batch size 1 with warmup runs discarded and many timed repeats (mean/std/median/
p95); **throughput** at a representative batch size; and **peak process RSS**
sampled during a real inference pass. `tracemalloc` is *not* used as a memory
metric, as it tracks only Python-level allocations and grossly understates true
footprint.

### 3.10 Reproducibility and Configuration

A single `seed_everything()` seeds Python, NumPy, and PyTorch and makes the
DataLoader deterministic (seeded generator + worker init). Every stage is a CLI
(`python -m src.<stage>`) driven by YAML configs with command-line overrides. The
end-to-end run is `make reproduce`.

**Reference environment.** Python 3.12; PyTorch 2.12; torchvision 0.27;
scikit-learn 1.9 (full pins in `requirements.txt`). Exact training hardware should
be recorded here from your run's `metadata.json` (GPU/CPU, RAM).

---

## Chapter 4: Results and Analysis

> All numbers below are filled by `scripts/render_report.py` from
> `results/metrics/*.json`. Full tables live in `results/tables/`.

### 4.1 Diagnostic Performance (Primary Model)

On the held-out test set (n = 624):

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| ROC-AUC | 0.9678 | [0.9504, 0.9816] |
| Accuracy | 0.9183 | [0.8958, 0.9391] |
| Sensitivity | 0.9667 | — |
| Specificity | 0.8376 | — |
| Precision | 0.9084 | — |
| F1-score | 0.9366 | — |

*Table 4 — Final Test Performance: `results/tables/table4_final_performance.md`.*

Interpretation should be written against the **measured** values once rendered
(e.g. whether the AUC objective of 0.95 is met, and the sensitivity/specificity
trade-off appropriate for screening). Do not assert targets are met before the
rendered numbers confirm it.

### 4.2 Model Comparison (Baselines)

EfficientNet-B0 is compared against ResNet-18 and MobileNetV3-Small under an
identical pipeline (parameters, AUC, accuracy, sensitivity, specificity, F1,
latency, size).

*Table 3 — Model Comparison: `results/tables/table3_model_comparison.md`.*

### 4.3 Quantization Results

Dynamic vs static INT8 (backend qnnpack):

- FP32 size: 17.67 MB.
- INT8 dynamic size: 16.69 MB (Linear layers only — modest reduction).
- INT8 static (PTQ) size: 5.22 MB
  (70.5% reduction), accuracy change 10.740%,
  AUC change 2.340%.

*Table 5 — Quantization Results: `results/tables/table5_quantization.md`.*
*Figure — `results/figures/quantization_comparison.png`.*

This contrast is itself a finding: on a convolution-heavy backbone, dynamic INT8
yields little compression, whereas static PTQ quantizes convolutions and delivers
the deployment-relevant reduction.

### 4.4 Computational Efficiency

On-disk size, latency (mean/p95), throughput, and peak RSS for each variant.

*Table 6 — Computational Efficiency: `results/tables/table6_efficiency.md`.*

FP32 reference latency: 119.70 ms/image; throughput
19.0 img/s (CPU). These CPU figures act as an edge proxy; actual
edge-hardware numbers are future work (Chapter 6).

### 4.5 Explainability

Grad-CAM++ overlays for correct, false-positive, and false-negative cases are in
`results/gradcam/`, with metadata in
`results/metrics/efficientnet_b0_gradcam_records.csv`. A composite figure is at
`results/figures/efficientnet_b0_gradcam_grid.png`. These are model-derived
visualisations for qualitative analysis only.

### 4.6 Error Analysis

Total errors: 51 of 624 (false positives 38, false
negatives 13); per-case details in
`results/metrics/efficientnet_b0_error_cases.csv`. Discuss the patterns observed in
the rendered errors (e.g. low-confidence predictions near the decision threshold).

---

## Chapter 5: Discussion

### 5.1 Summary

Once rendered, summarise the measured trade-offs across the three axes:
performance (Section 4.1–4.2), efficiency including the dynamic-vs-static
quantization contrast (4.3–4.4), and qualitative explainability (4.5).

### 5.2 Clinical Implications

The framework is positioned as a **screening / decision-support** tool, not a
standalone diagnostic. Probability outputs plus Grad-CAM++ overlays support a
human-in-the-loop workflow in which low-confidence cases are flagged for expert
review. The Youden-J threshold allows tuning the sensitivity/specificity balance
to the clinical context.

### 5.3 Limitations

This work states its limitations explicitly (see `reports/limitations.md`):
single-dataset evaluation; no external validation; no real-world/clinician
validation (and no simulated validation anywhere); no edge-hardware benchmarking;
and potential dataset bias (pediatric, class-imbalanced, single-source).

### 5.4 Comparison with Prior Work

Use the **literature comparison template**
(`results/tables/literature_comparison_template.md`) to position this work against
prior studies (author, year, dataset, model, accuracy, AUC, explainability,
quantization, edge focus). Literature values must be cited from their sources and
clearly labelled as literature — never mixed with measured results.

### 5.5 Ethical Considerations

Research/decision-support use only; not a medical device. Public, de-identified
data were used. Subgroup fairness (age, sex, ethnicity) was not assessed and is
future work. Clinical deployment would require regulatory approval and prospective
validation.

---

## Chapter 6: Conclusion and Future Work

### 6.1 Conclusion

This thesis delivers a reproducible, explainable, and quantization-aware pneumonia
detection framework with rigorous, correctly measured efficiency reporting.
Headline measured outcomes (fill from rendered results): test AUC 0.9678,
static-INT8 size 5.22 MB (70.5% smaller than
FP32).

### 6.2 Future Work

See `reports/future_work.md`: external validation (NIH ChestX-ray14), CheXpert
validation, edge-hardware deployment (Raspberry Pi / Jetson), TinyML optimisation
(QAT, pruning, distillation), federated learning, and multi-disease extension.
These extend naturally into a PhD research programme on trustworthy, resource-
efficient medical AI.

### 6.3 Final Remarks

High diagnostic value need not require high-resource infrastructure — but the
claim must be earned through honest measurement. By releasing a fully reproducible
pipeline that refuses to fabricate results, this work aims to support responsible
translation of medical AI to underserved settings.

---

## References

1. Walker CLF, et al. Global burden of childhood pneumonia and diarrhoea. *Lancet*. 2013;381(9875):1405-1416.
2. World Health Organization. Pneumonia (fact sheet). 2021.
3. Callahan A, et al. Artificial intelligence in global health. *NPJ Digit Med*. 2021;4:34.
4. Rajpurkar P, et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv:1711.05225*. 2017.
5. Topol EJ. High-performance medicine. *Nat Med*. 2019;25(1):44-56.
6. Esteva A, et al. A guide to deep learning in healthcare. *Nat Med*. 2019;25(1):24-29.
7. Amann J, et al. Explainability for AI in healthcare. *BMC Med Inform Decis Mak*. 2020;20:310.
8. Deng J, et al. ImageNet: A large-scale hierarchical image database. *CVPR*. 2009.
9. Wang X, et al. ChestX-ray8. *CVPR*. 2017.
10. Irvin J, et al. CheXpert. *AAAI*. 2019.
11. Chaves J, et al. Self-supervised learning for medical image analysis: a review. *Med Image Anal*. 2023;84:102714.
12. Sandler M, et al. MobileNetV2. *CVPR*. 2018.
13. Tan M, Le Q. EfficientNet. *ICML*. 2019.
14. Wang G, et al. Lightweight Deep Learning for Pneumonia Detection on Chest X-rays. *IEEE JBHI*. 2022;26(5):2105-2114.
15. Zhou B, et al. Learning Deep Features for Discriminative Localization. *CVPR*. 2016.
16. Selvaraju RR, et al. Grad-CAM. *ICCV*. 2017.
17. Chattopadhay A, et al. Grad-CAM++. *WACV*. 2018.
18. Saporta A, et al. Benchmarking saliency methods for chest X-ray interpretation. *Nat Mach Intell*. 2022;4:867-878.
19. Tonekaboni S, et al. What Clinicians Want. *arXiv:2108.04838*. 2021.
20. Jacob B, et al. Quantization and Training of Neural Networks for Efficient Inference. *CVPR*. 2018.
21. David R, et al. TensorFlow Lite Micro. *MLSys*. 2021.
22. Kermany DS, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*. 2018;172(5):1122-1131.e9.
23. Pepe MS. The Statistical Evaluation of Medical Tests for Classification and Prediction. Oxford University Press; 2003.

---

## Appendices

### Appendix A: Reproducibility

```bash
make setup        # create venv + install pinned dependencies
make validate-data
make reproduce    # train (3 models) -> evaluate -> benchmark -> quantize -> explain -> report -> render
```

Each run writes a `metadata.json` manifest (timestamp, git SHA, seed, library
versions, config, dataset stats). Seeds are fixed; data loading is deterministic.

### Appendix B: Repository Structure

See `README.md` for the full layout (`src/`, `configs/`, `tests/`, `results/`,
`reports/`, `paper_assets/`).

### Appendix C: Ethics Statement

Public, de-identified Kermany data (CC BY 4.0) were used. This software is a
research artifact, not a medical device, and is not approved for diagnostic use.
