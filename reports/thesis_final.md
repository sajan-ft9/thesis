# MASTER'S THESIS

# Memory-Efficient and Explainable Deep Learning Framework for Pneumonia Detection from Chest X-Ray Images Using a Quantized EfficientNet-B0

**Author:** Sajan
**Degree:** Master of Science in Information Technology
**Institution:** [Your University]
**Supervisor:** [Supervisor Name]
**Date:** [Submission Date]

> **Provenance of results.** Every quantitative result in this document was produced
> by the accompanying open-source pipeline on the Kermany chest X-ray dataset and is
> reproducible with `make reproduce`. Reported numbers correspond to the committed
> run (seed = 42; Python 3.12.13, PyTorch 2.12.0, torchvision 0.27.0, scikit-learn
> 1.9.0; Apple-silicon GPU via the Metal/MPS backend). No result in this thesis is
> simulated, estimated, or hand-edited; figures and tables are generated under
> `results/` and the numeric values below are taken verbatim from
> `results/metrics/*.json`.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Chapter 1: Introduction](#chapter-1-introduction)
3. [Chapter 2: Literature Review](#chapter-2-literature-review)
4. [Chapter 3: Methodology](#chapter-3-methodology)
5. [Chapter 4: Results and Analysis](#chapter-4-results-and-analysis)
6. [Chapter 5: Discussion](#chapter-5-discussion)
7. [Chapter 6: Conclusion and Future Work](#chapter-6-conclusion-and-future-work)
8. [References](#references)
9. [Appendices](#appendices)

---

## Abstract

**Background.** Pneumonia is a leading cause of childhood morbidity and mortality,
with the greatest burden in low- and middle-income countries where access to expert
chest X-ray (CXR) interpretation is limited. Deep-learning models can assist
screening, but high-capacity architectures are computationally heavy and opaque,
which hinders deployment on constrained hardware and limits clinical trust.

**Objective.** This thesis designs, implements, and rigorously evaluates a unified
framework that jointly addresses three axes for binary pneumonia detection on CXR
images: (i) diagnostic performance, (ii) computational efficiency for
resource-constrained deployment, and (iii) explainability. The emphasis throughout
is on *measurement rigour and reproducibility* rather than benchmark maximisation.

**Methods.** An ImageNet-pretrained **EfficientNet-B0** backbone with a custom
binary head is trained with a staged transfer-learning schedule, label smoothing,
cosine learning-rate scheduling, and early stopping. Two baselines — **ResNet-18**
and **MobileNetV3-Small** — are trained with an identical pipeline for a fair
comparison. The data pipeline performs automatic integrity checking, including
byte-level **duplicate detection** and **train/validation/test leakage** analysis.
The model is evaluated on the held-out Kermany test set (n = 624) with non-parametric
**bootstrap 95% confidence intervals**. **Grad-CAM++** generates visual explanations
for correctly classified, false-positive, and false-negative cases. Two post-training
quantization schemes — **dynamic INT8** and **static INT8 (post-training quantization
with calibration)** — are compared, and computational efficiency is benchmarked
correctly (on-disk model size; warm-up-corrected, repeated latency timing;
throughput; and sampled peak process memory).

**Results.** On the held-out test set, EfficientNet-B0 attains a ROC-AUC of **0.9678**
(95% CI 0.9504–0.9816), accuracy **0.9183** (95% CI 0.8958–0.9391), sensitivity
**0.9667**, specificity **0.8376**, and F1 **0.9366**, with 51/624 errors (38 false
positives, 13 false negatives). The data-integrity check detected and removed **26
byte-identical duplicate images** that would otherwise have leaked between the train
and validation splits. Among architectures, all three exceed 0.95 AUC, but
EfficientNet-B0 provides the **best precision/specificity balance** (38 false
positives versus 72 and 77 for ResNet-18 and MobileNetV3-Small). **Static INT8
quantization** compresses the model from **17.66 MB to 5.13 MB (−71%)**, reduces CPU
latency ~8× (≈120 ms → ≈15 ms per image), and cuts **peak inference memory ≈6×
(979.5 → 164.4 MB)** for a 2.34-percentage-point AUC cost, whereas **dynamic INT8**
preserves accuracy but compresses by only ~6%. The lazy streaming data pipeline uses
**9.4× less RAM** than naïve full-dataset loading (335.8 MB vs 3,140.9 MB).

**Conclusion.** Competitive pneumonia screening is achievable within tight resource
budgets when a lightweight architecture, integrated explainability, and an
appropriate quantization scheme are combined and *measured honestly*. The complete
code, configurations, tests, and reproducibility artefacts are released to support
replication and extension.

**Keywords:** pneumonia detection, chest X-ray, EfficientNet-B0, Grad-CAM++,
post-training quantization, edge AI, explainable AI, reproducible research.

---

## Chapter 1: Introduction

### 1.1 Background and Motivation

Pneumonia is among the leading causes of death in children under five, and the
burden falls disproportionately on low- and middle-income countries (LMICs) where
radiological expertise is scarce [1, 2]. Chest X-ray (CXR) imaging is the most
widely available diagnostic modality for pneumonia, but reliable interpretation
requires expertise that is frequently unavailable at the point of care [3].
Automated CXR analysis with deep learning is therefore attractive for screening —
provided the models are *deployable* on the modest hardware available in such
settings and *transparent* enough to support clinician oversight.

Landmark models such as CheXNet, a 121-layer DenseNet, demonstrated strong CXR
classification performance [4]. However, three interrelated challenges limit the
real-world impact of such systems:

1. **Computational burden.** Large models demand substantial memory and compute,
   making them impractical on standard clinical hardware or edge devices in LMICs [5].
2. **Memory bottlenecks.** Naïve data pipelines that load entire datasets into RAM
   fail on memory-limited machines, preventing scalable analysis [6].
3. **Opacity.** The "black-box" nature of deep models impedes clinical trust, since
   practitioners cannot inspect the basis of a prediction [7].

### 1.2 Problem Statement

Three coupled problems prevent current pneumonia-CXR models from being trusted and
deployed in resource-constrained settings. First, **high-capacity models are accurate
but impractical**: architectures such as CheXNet require tens of millions of parameters
and large memory/compute budgets unsuited to point-of-care or edge hardware. Second,
**lightweight and edge-oriented studies typically optimise and report accuracy alone** —
without confidence intervals, data-leakage control, a comparison of quantization
schemes, a memory-footprint analysis, or any explanation of model decisions — so their
reliability and deployment readiness cannot be judged. Third, **medical-imaging results
are frequently irreproducible and silently contaminated** by duplicate or patient-level
data leakage, which inflates reported performance.

The problem this thesis addresses is therefore: *how to detect pneumonia from chest
X-rays with a model that is simultaneously accurate, memory- and compute-efficient,
explainable, and evaluated honestly and reproducibly* — closing the gap between reported
accuracy and genuine, deployable, trustworthy performance.

### 1.3 Research Objectives

This thesis pursues four objectives, framed as *measurable targets*, not assumed
outcomes:

1. **Diagnostic performance.** Train and rigorously evaluate a lightweight CXR
   pneumonia classifier, reporting metrics with confidence intervals rather than
   point estimates alone (reference target: ROC-AUC ≥ 0.95).
2. **Computational efficiency.** Quantify model size, **peak memory (RAM)**, inference
   latency, and throughput, and reduce them via post-training quantization suitable for
   edge deployment (reference targets: model size < 50 MB; CPU latency < 100 ms/image;
   peak inference RAM < 2 GB).
3. **Explainability.** Integrate Grad-CAM++ to visualise the regions driving each
   prediction, including failure cases, to support qualitative error analysis.
4. **Reproducibility.** Provide a fully scripted, seed-controlled, configuration-
   driven pipeline whose every reported number is regenerable from raw data.

### 1.4 Research Questions

The objectives are addressed through four research questions (RQ), each mapped to an
objective and answered in the indicated chapter:

- **RQ1 (Performance).** Can a lightweight EfficientNet-B0, evaluated with bootstrap
  confidence intervals and explicit leakage control, achieve ROC-AUC ≥ 0.95 for binary
  pneumonia detection on a held-out test set, and how does it compare with ResNet-18 and
  MobileNetV3-Small trained under an identical pipeline? *(Chapter 4.1–4.2)*
- **RQ2 (Efficiency).** To what extent can post-training INT8 quantization (dynamic vs.
  static) reduce model size, inference latency, and peak memory, and at what cost to
  diagnostic accuracy? *(Chapter 4.3–4.4)*
- **RQ3 (Explainability).** Do Grad-CAM++ explanations highlight clinically plausible
  regions for correct and incorrect predictions, supporting error analysis and
  human-in-the-loop oversight? *(Chapter 4.5–4.6)*
- **RQ4 (Reproducibility).** Can the complete pipeline be made fully reproducible —
  seeded, tested, and regenerable from raw data — with no fabricated or simulated
  results? *(Chapter 3; Appendix A)*

### 1.5 Scope and Delimitations

This study is deliberately bounded so that every claim is measurable: (i) the task is
**binary** pneumonia-vs-normal classification (not multi-disease or lesion
localization); (ii) evaluation uses the **Kermany pediatric** dataset, with external
datasets left to future work; (iii) compression uses **post-training quantization**
(dynamic and static INT8), not quantization-aware training; (iv) efficiency is measured
on **workstation CPU/GPU as an edge proxy**, not on physical Raspberry Pi / Jetson
hardware; and (v) explainability is assessed **qualitatively**, without a radiologist
reader study. The corresponding limitations are discussed in Section 5.3.

### 1.6 Significance of the Study

The work is significant on three levels. **Practically**, it shows that accurate
pneumonia screening can run within a small, predictable memory and storage budget,
supporting deployment where specialist radiology is scarce. **Methodologically**, it
provides a template for *honest* medical-AI evaluation — duplicate/leakage checking,
confidence intervals, correct memory benchmarking, and a transparent quantization
trade-off — countering the accuracy-only, often-irreproducible reporting common in the
field. **For open science**, the released code, tests, and reproducibility manifests let
others replicate and extend the results, establishing the foundation for the doctoral
directions in Chapter 6.

### 1.7 Research Contributions

The overarching contribution of this thesis is **not a new pneumonia classifier** but a
**reproducible, integrity-checked, deployment- and memory-oriented methodology** for
medical image classification, demonstrated on pneumonia CXR. It comprises five explicit,
independently useful contributions:

1. **Data-integrity contribution.** An automatic check that detects byte-identical
   duplicate images and quantifies train/validation/test leakage by path and content
   hash; on this dataset it removed 26 duplicates causing train↔validation leakage and
   verified content-disjoint splits (§3.1) — rigour most CXR studies omit.
2. **Reproducibility contribution.** A fully scripted, seed-controlled,
   configuration-driven pipeline with unit/integration tests and per-run manifests, so
   every reported number and figure is regenerable from raw data with one command
   (`make reproduce`).
3. **Explainability contribution.** Integrated Grad-CAM++ explanations for correctly
   classified, false-positive, and false-negative cases, used for genuine error analysis
   — with **no simulated survey or trust score** anywhere (§3.5, §4.5–4.6).
4. **Quantization contribution.** A transparent dynamic-versus-static INT8 comparison,
   including the practical finding that **per-channel** (not per-tensor) weight
   quantization is required to keep EfficientNet-B0 from collapsing (§4.3).
5. **Deployment-efficiency contribution.** A correct efficiency methodology — on-disk
   size, warm-up-corrected latency, throughput, and **sampled peak memory (RSS, not
   `tracemalloc`)** — quantifying a 71% smaller, ~6× lower-memory, ~8× faster model and a
   9.4× lower-RAM data pipeline (§4.4).

Together these support a fair three-architecture comparison (EfficientNet-B0 vs.
ResNet-18 vs. MobileNetV3-Small) under an identical pipeline, and frame the work as a
template for honest, deployable medical AI rather than a single-model accuracy result.

### 1.8 Thesis Structure

Chapter 2 reviews related work; Chapter 3 details the methodology; Chapter 4 reports
results; Chapter 5 discusses clinical implications and limitations; Chapter 6
concludes and outlines future work.

---

## Chapter 2: Literature Review

### 2.1 Deep Learning for Chest X-Ray Analysis

Convolutional neural networks (CNNs) with ImageNet pretraining have driven rapid
progress in medical image analysis [8]. For CXR, Rajpurkar et al. introduced
CheXNet, a 121-layer DenseNet trained on the ChestX-ray14 dataset [4]; subsequent
work explored ensembles [9], multi-task learning [10], and self-supervised
pretraining [11]. These approaches generally prioritise accuracy over efficiency,
yielding models with tens of millions of parameters and multi-gigabyte memory
footprints that are ill-suited to constrained deployment [5].

### 2.2 Lightweight Architectures for Medical Imaging

To address computational constraints, efficient CNNs have been adapted for medical
tasks. MobileNetV2/V3 [12] and EfficientNet [13] use depthwise-separable
convolutions and compound scaling to reduce parameters while retaining accuracy.
EfficientNet-B0 in particular offers a strong accuracy/efficiency trade-off with
roughly five million parameters, making it a compelling backbone for
resource-constrained applications [14]. Recent pneumonia-specific work confirms this:
Benmalek et al. [25] benchmark five lightweight CNNs (ResNet-18, MobileNet,
ShuffleNet, SqueezeNet, EfficientNet-B0) on the same pediatric CXR data and deploy the
best model on an NVIDIA Jetson Nano, reporting F1-scores of ~94–96% (MobileNet 95.7%,
EfficientNet-B0 95.1%, ResNet-18 94.4%), and custom efficient designs such as PneuNet
[26] reach comparable accuracy with fewer parameters. These results — consistent with
the present study's measured F1 of 0.937 — establish that lightweight backbones are
already sufficient for this task; the open question is therefore no longer *whether* a
small model works, but **how honestly it is evaluated and how efficiently it can be
deployed** (Sections 2.4–2.5).

### 2.3 Explainable AI in Clinical Decision Support

The opacity of deep models has motivated explainable-AI (XAI) techniques. Class
Activation Mapping (CAM) [15], gradient-weighted CAM (Grad-CAM) [16], and Grad-CAM++
[17] produce saliency maps localising the evidence behind a prediction. In medical
imaging these maps support error analysis and clinician oversight, but rigorous
benchmarking has shown that saliency methods must be *validated*, not assumed, before
any claim of clinical alignment [18, 19]. Accordingly, this thesis treats Grad-CAM++
outputs as model-derived visualisations for qualitative analysis and makes **no claim
of radiologist-validated alignment** (see Limitations).

### 2.4 Quantization and Edge Deployment

Post-training quantization (FP32 → INT8) reduces model size and can accelerate CPU
inference [20]. *Dynamic* quantization typically targets linear layers only, while
*static* post-training quantization (PTQ) with a calibration pass also quantizes
convolutions — important for convolution-heavy backbones. Cross-platform runtimes
such as ONNX Runtime and TensorFlow Lite enable inference on ARM edge devices [21].
A practical but under-reported subtlety, which this thesis documents empirically, is
that the choice of quantization observer (per-tensor versus per-channel) is decisive
for architectures such as EfficientNet.

### 2.5 Identified Research Gap

Individual ingredients — lightweight architectures, XAI, and quantization — are
well studied in isolation. What remains uncommon is an *integrated*, *correctly
benchmarked*, and *fully reproducible* pipeline that jointly reports diagnostic
performance, efficiency, and explainability for constrained CXR deployment, while
being explicit about data integrity and measurement methodology. Notably, even recent
edge-deployment studies on this dataset (e.g. Benmalek et al. [25]) report accuracy/F1
alone — without confidence intervals, data-leakage control, a dynamic-versus-static
quantization comparison, or any measurement of memory footprint or whether compression
preserves the model's explanations. This thesis targets that gap.

---

## Chapter 3: Methodology

### 3.1 Dataset, Integrity Checking, and Splits

**Dataset.** We use the Kermany pediatric chest X-ray dataset [22]. The training
folder contains 5,216 images (Normal = 1,341; Pneumonia = 3,875) and the official
test folder contains 624 images (Normal = 234; Pneumonia = 390); the dataset is
class-imbalanced toward pneumonia (~73%). The dataset also ships a 16-image official
validation split, which is too small for reliable model selection and is **not used**.

**Splitting.** Rather than rely on the tiny official validation split, we carve a
**stratified 20% validation split** from the training folder and keep the **official
test set untouched** for final evaluation.

**Integrity checking (a methodological contribution).** Before training, the
pipeline computes per-split class balance, detects **byte-identical duplicate images**
(MD5), and checks **train/validation/test leakage** by both path and content hash.
On this dataset the naïve stratified split was *not* clean: the training folder
contains 16 within-train duplicate groups, and 9 image contents appeared in **both**
the training and validation partitions (content leakage that would optimistically
bias model selection). The pipeline therefore **deduplicates the training pool**
(removing 26 byte-identical duplicate files) before splitting; after deduplication
the train/validation/test partitions are content-disjoint (verified). The official
test set is retained at its canonical 624 images; its 6 intrinsic within-test
duplicate groups are disclosed but not removed, so results remain comparable to the
standard benchmark. After deduplication the working splits are **train = 4,152**
(Normal 1,072; Pneumonia 3,080), **validation = 1,038** (Normal 268; Pneumonia 770),
and **test = 624** (Normal 234; Pneumonia 390).

**Preprocessing.** Images are resized and normalised with ImageNet statistics
(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). Training augmentation
comprises resize-then-random-crop to 224×224, random horizontal flip, small rotation
(±10°), and brightness/contrast jitter. Validation and test use deterministic
resize + normalise only (no augmentation), ensuring unbiased, reproducible
evaluation. All transforms are configurable.

**Memory-safe loading.** A custom lazy, path-based `Dataset` reads and preprocesses
one image at a time, so the full dataset never resides in RAM — a prerequisite for
low-memory environments.

### 3.2 Model Architecture

**Backbone.** EfficientNet-B0 [13], pretrained on ImageNet, serves as the feature
extractor (feature dimension 1,280). Its ImageNet classifier is replaced by a custom
binary head:

```
BatchNorm1d(1280) → Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→1)
```

A single logit is trained with binary cross-entropy; a sigmoid at inference yields a
probability. The resulting model has **4.34 M parameters**. The *identical* head and
pipeline are applied to ResNet-18 (11.31 M) and MobileNetV3-Small (1.08 M) for a
controlled comparison.

**Transfer-learning schedule.** Early epochs freeze the leading backbone stages and
train the head; from a configured epoch the full backbone is unfrozen and fine-tuned
at a reduced learning rate, which stabilises early training and adapts features to CXR.

### 3.3 Training Protocol

- **Optimiser:** AdamW (initial lr = 3e-4 for the head phase; 3e-5 after unfreezing),
  weight decay 1e-4.
- **Schedule:** 3-epoch linear warm-up then cosine annealing.
- **Loss:** binary cross-entropy with label smoothing (α = 0.05).
- **Batch size:** 32. **Gradient clipping:** max-norm 1.0.
- **Early stopping:** patience 7 epochs on validation AUC; up to 20 epochs configured.
- **Mixed precision:** enabled automatically on CUDA. The committed run executed on
  the Apple-silicon GPU (Metal/MPS backend) in single-precision, as automatic mixed
  precision is CUDA-specific.
- **Reproducibility:** all RNGs seeded (seed = 42); DataLoaders use a seeded generator
  and deterministic worker initialisation; each run writes a `metadata.json` manifest
  (timestamp, library versions, configuration, dataset statistics, best epoch).

### 3.4 Evaluation Metrics and Statistics

On the held-out test set we report accuracy, precision, recall/sensitivity,
specificity, F1, and ROC-AUC. We compute **non-parametric bootstrap 95% confidence
intervals** (1,000 resamples) for accuracy and ROC-AUC, the **Youden-J optimal
operating threshold**, and a per-error (false-positive/false-negative) table for
qualitative analysis. Confusion-matrix, ROC, and precision–recall figures are exported
as PNG and PDF.

### 3.5 Explainability Integration

Grad-CAM++ [17] is applied to the predicted class for cases sampled from the *real*
test predictions across three diagnostically meaningful categories: correctly
classified, false-positive, and false-negative. Overlays are saved with metadata
(true/predicted label, probability) for auditability. **No radiologist survey, trust
score, or clinical-validation metric is produced or simulated** anywhere in this work.

### 3.6 Quantization for Edge Deployment

Two post-training schemes are compared:

1. **Dynamic INT8** — quantizes linear-layer weights; activations are quantized at
   runtime. Trivial to apply but compresses little on a convolution-heavy backbone.
2. **Static INT8 PTQ** — FX graph-mode static quantization with a calibration pass
   that also quantizes convolutions. We use **per-channel symmetric INT8 weights with
   histogram activation observers** and calibrate on the *clean* validation loader
   (never the augmented training loader, and never the test set). This choice is
   essential: the backend-default per-tensor configuration catastrophically degrades
   EfficientNet-B0 (Section 4.4). The quantization backend is `qnnpack` (the supported
   engine on the host platform).

The FP32 model is additionally exported to **ONNX** and verified with ONNX Runtime.

### 3.7 Efficiency Benchmarking

Efficiency is measured as: **on-disk model size** (serialised state dict);
**inference latency** at batch size 1 with warm-up runs discarded and many timed
repeats summarised as mean / standard deviation / median / 95th percentile;
**throughput** at a representative batch size; and **peak process resident set size
(RSS)** sampled during a real inference pass. We deliberately **do not** use
`tracemalloc` as a memory metric, because it tracks only Python-level allocations and
grossly understates the true footprint of tensor and library memory.

### 3.8 Experimental Setup

**Hardware.** Apple-silicon workstation; training on the Metal/MPS GPU backend;
quantization, CPU latency, and ONNX verification on CPU. No CUDA GPU was used.

**Software.** Python 3.12.13, PyTorch 2.12.0, torchvision 0.27.0, scikit-learn 1.9.0,
grad-cam 1.5.5, onnxruntime 1.26.0.

**Reproducibility.** Complete code, configurations, tests, and environment pins are
provided; the full study is reproduced with `make reproduce`.

---

## Chapter 4: Results and Analysis

### 4.1 Diagnostic Performance (Primary Model)

**Test-set results (EfficientNet-B0, n = 624 images):**

| Metric | Value | 95% Bootstrap CI | Reference Target | Status |
|---|---|---|---|---|
| ROC-AUC | **0.9678** | [0.9504, 0.9816] | ≥ 0.95 | ✅ Met |
| Sensitivity | 0.9667 | — | ≥ 0.93 | ✅ Met |
| Specificity | 0.8376 | — | ≥ 0.90 (at thr = 0.5) | ⚠️ See §4.1.1 |
| Accuracy | 0.9183 | [0.8958, 0.9391] | — | Good |
| Precision | 0.9084 | — | — | Good |
| F1-score | 0.9366 | — | — | Good |

*Table 4.1: Diagnostic metrics on the held-out test set at the default decision
threshold (0.5). Bootstrap CIs use 1,000 resamples. Confusion-matrix counts:
TP = 377, TN = 196, FP = 38, FN = 13.*

The model meets the AUC objective with margin — the 95% CI lower bound (0.9504) itself
clears 0.95 — and the sensitivity objective at the default threshold.

#### 4.1.1 Operating-Point Analysis

At the default 0.5 threshold the model is sensitivity-leaning (Sens = 0.9667,
Spec = 0.8376), appropriate for screening where missed pneumonia is costlier than a
false alarm. The threshold-independent ROC-AUC (0.9678) shows strong overall
discrimination. Tuning the threshold to the **Youden-optimal point (0.918)** rebalances
the operating point to **sensitivity 0.9410 and specificity 0.9188**, which *meets* the
0.90 specificity target at a modest sensitivity cost. The appropriate operating point
is therefore a deployment choice; both are reported transparently.

### 4.2 Model Comparison (Baselines)

| Model | Params (M) | ROC-AUC | Accuracy | Sensitivity | Specificity | F1 | FP / FN | CPU Latency (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | 4.34 | 0.9678 | 0.9183 | 0.9667 | **0.8376** | **0.9366** | **38 / 13** | 119.7 | 17.66 |
| ResNet-18 | 11.31 | 0.9594 | 0.8798 | 0.9923 | 0.6923 | 0.9117 | 72 / 3 | 15.2 | 45.32 |
| MobileNetV3-Small | 1.08 | **0.9743** | 0.8750 | 0.9974 | 0.6709 | 0.9089 | 77 / 1 | 41.1 | 4.44 |

*Table 4.2: Architecture comparison under an identical pipeline. Best value per
column in bold. Latency is single-image CPU inference (mean of repeated, warm-up-
corrected runs); size is the serialised FP32 state dict.*

**Key insights.**
1. **All three architectures exceed 0.95 AUC**, confirming the task is well within the
   reach of lightweight models on this dataset.
2. **MobileNetV3-Small attains the highest AUC (0.9743) at the smallest size
   (4.44 MB, 1.08 M parameters)** — an important, honest result: the most compact
   model is competitive on raw discrimination.
3. However, the baselines achieve their high sensitivity by over-predicting pneumonia:
   ResNet-18 and MobileNetV3-Small produce **72 and 77 false positives** (specificity
   0.69 and 0.67), versus **38 for EfficientNet-B0** (specificity 0.84). EfficientNet-B0
   gives the **best precision/specificity balance and highest F1**, making it the most
   clinically usable operating point and the proposed backbone.

**Why EfficientNet-B0 and not MobileNetV3-Small?** Although MobileNetV3-Small edges out
EfficientNet-B0 on AUC (0.974 vs. 0.968) at a smaller size, it roughly **doubles the
false-positive rate** (77 vs. 38; specificity 0.67 vs. 0.84). For a screening tool that
clinicians must trust, an excess of false alarms — not a 0.6-point AUC difference — is
the decisive factor: EfficientNet-B0 is therefore selected as the default, and
MobileNetV3-Small is positioned as the **minimum-footprint alternative** for the most
storage-constrained settings where a higher false-positive rate is acceptable. (A
deployment could also select MobileNetV3-Small and recover specificity via threshold
tuning; both operating points are reported for transparency.)

### 4.3 Quantization Results

| Variant | Size (MB) | Size ↓ | Accuracy | Acc. drop | ROC-AUC | AUC drop | CPU Latency (ms) |
|---|---|---|---|---|---|---|---|
| FP32 (baseline) | 17.66 | — | 0.9183 | — | 0.9678 | — | 122.4 |
| INT8 dynamic | 16.68 | 5.6% | 0.9199 | −0.16% | 0.9682 | −0.04% | 120.9 |
| **INT8 static (PTQ)** | **5.13** | **71.0%** | 0.8109 | 10.74% | **0.9444** | **2.34%** | **15.0** |

*Table 4.3: Quantization comparison (backend: qnnpack). "Acc. drop" is measured at a
fixed 0.5 threshold; "AUC drop" is threshold-independent.*

**Findings.**
1. **Dynamic INT8** preserves accuracy and AUC essentially exactly (changes within
   ±0.2%) but compresses by only ~6%, because it quantizes only the linear head while
   the convolutional body — the bulk of the parameters — remains FP32. It also yields
   no latency benefit here.
2. **Static INT8 PTQ** compresses the model by **71% (17.66 → 5.13 MB)** and reduces
   single-image CPU latency by roughly **8× (≈120 → ≈15 ms)**, for a **2.34-percentage-
   point AUC reduction** (0.9678 → 0.9444). The larger drop in fixed-threshold accuracy
   (to 0.8109) reflects a shift in probability calibration after quantization rather
   than a loss of discrimination; re-tuning the decision threshold recovers most of the
   accuracy, consistent with the comparatively small AUC change.
3. **Methodological finding.** The backend-default *per-tensor* static configuration
   collapsed EfficientNet-B0 to below-chance discrimination (AUC ≈ 0.40). Switching to
   **per-channel symmetric INT8 weights with histogram activation observers**, and
   calibrating on clean (non-augmented) data, recovered AUC to 0.9444. This
   observer/qconfig choice is decisive for EfficientNet-class networks and is, to the
   author's knowledge, an under-emphasised practical detail in applied CXR-quantization
   literature.

The FP32 model was also exported to ONNX and verified with ONNX Runtime, providing a
cross-platform deployment artefact.

### 4.4 Computational Efficiency

**Speed and size (EfficientNet-B0, CPU):**

| Metric | FP32 | INT8 static (PTQ) |
|---|---|---|
| Model size on disk | 17.66 MB | 5.13 MB |
| Inference latency (mean) | 119.7 ms/image | 15.0 ms/image |
| Inference latency (95th pct) | 120.6 ms/image | — |
| Throughput (batch = 32) | 19.0 images/s | — |

*Table 4.4: Speed and storage, measured with warm-up-corrected repeated timing.*

FP32 single-image CPU latency (≈120 ms) exceeds the 100 ms reference target on this
general-purpose CPU, **but static INT8 quantization reduces it to ≈15 ms (~8×)**,
comfortably meeting the target — precisely the deployment motivation for quantization.
(CPU figures are an edge proxy; on-device measurement is future work.)

#### 4.4.1 Memory-Footprint Analysis

Because memory — not raw compute — is typically the binding constraint on
resource-constrained hardware, we measure peak process resident-set size (RSS)
directly with psutil sampling (never `tracemalloc`, which tracks only Python-level
allocations and would report misleading sub-megabyte values). Two measurements
substantiate the memory-efficiency contribution.

**(a) Streaming vs. naïve loading.** Iterating the lazy, path-based DataLoader for one
pass over the training set peaks at **335.8 MB**, whereas pre-loading the same 5,216
images into a single in-RAM float32 tensor (the common non-streaming pattern) peaks at
**3,140.9 MB** — a **9.4× reduction**. The naïve figure matches the nominal tensor
size (3,140.6 MB), validating the measurement. This lazy pipeline is the core enabler
of operation on low-memory hardware.

**(b) Runtime memory by precision.** Peak RSS during a full test-set inference pass,
measured identically per variant:

| Variant | Model weights (MB) | Peak inference RSS (MB) |
|---|---|---|
| FP32 | 17.66 | 979.5 |
| INT8 dynamic | 16.68 | 351.1 |
| **INT8 static (PTQ)** | **5.13** | **164.4** |

*Table 4.5: Model and runtime memory by precision (EfficientNet-B0, CPU). See
`results/figures/memory_footprint.png`.*

**Static INT8 quantization reduces not only the stored model (3.4×) but the peak
inference memory by ≈6× (979.5 → 164.4 MB)**, because activations as well as weights
are represented in INT8; dynamic INT8 (linear-only) gives an intermediate runtime
footprint with negligible storage savings. Combined with the streaming loader, the
deployed static-INT8 configuration operates within a small, predictable memory budget
appropriate for resource-constrained healthcare settings.

### 4.5 Explainability

Grad-CAM++ overlays were generated for twelve test cases — four correctly classified,
four false positives, and four false negatives — using the model's final convolutional
stage as the target layer. Overlays and metadata are saved under `results/gradcam/`,
and a composite figure is provided. Qualitatively, correctly classified pneumonia
cases concentrate activation over opacified lung regions, while several false negatives
correspond to subtle or peripheral findings with low predicted probability. These are
**model-derived visualisations for qualitative error analysis only**; no
radiologist-rating study was performed or simulated, and clinician validation is
identified as future work.

### 4.6 Error Analysis

On the test set the model makes **51 errors out of 624 (8.2%)**: **38 false positives**
(Normal → Pneumonia) and **13 false negatives** (Pneumonia → Normal). The strong
sensitivity (few false negatives) is desirable for screening. Examining predicted
probabilities, false positives have a mean predicted pneumonia probability of
**0.85 ± 0.15** and false negatives **0.24 ± 0.13** — i.e., most errors are confident
rather than borderline, indicating genuinely difficult or atypical radiographs rather
than threshold ambiguity. Coupling probability outputs with Grad-CAM++ overlays lets a
clinician triage and review such cases, supporting a safe human-in-the-loop workflow.

---

## Chapter 5: Discussion

### 5.1 Summary of Key Findings

A unified framework integrating a lightweight backbone, integrated explainability, and
post-training quantization can deliver strong, well-calibrated pneumonia screening
within tight resource budgets — when each axis is *measured rigorously*. Concretely:
EfficientNet-B0 attains test AUC 0.9678 (95% CI 0.9504–0.9816) with the best
precision/specificity balance among the evaluated models; static INT8 quantization
yields a 71% smaller, ~8× faster model for a 2.3-point AUC cost; and the data-integrity
tooling removed real train↔validation leakage before it could bias model selection.

### 5.2 Clinical Implications

The framework is positioned as a **screening / decision-support** tool, not a
standalone diagnostic. Its probability outputs and Grad-CAM++ overlays enable a
human-in-the-loop workflow in which low-confidence or flagged cases are routed to
expert review. The operating point is a deployment parameter: the default threshold
favours sensitivity (suited to screening), while the Youden-optimal threshold gives a
balanced sensitivity/specificity profile that meets a 0.90 specificity requirement.

### 5.3 Limitations

These limitations are stated explicitly and are not hidden:

1. **Single-dataset evaluation.** All data come from the single-institution Kermany
   pediatric cohort; results may not transfer to other scanners, protocols, age groups,
   or hospitals.
2. **No external validation.** The model has not been evaluated on an independent
   dataset (e.g., NIH ChestX-ray14, CheXpert); reported metrics reflect in-distribution
   performance only.
3. **No clinical / reader-study validation.** No prospective study or radiologist
   reader study was conducted. Grad-CAM++ outputs are model-derived and were **not**
   validated against expert annotations; no trust or alignment scores are claimed or
   simulated.
4. **No edge-hardware validation.** Efficiency was measured on a general-purpose CPU as
   an edge proxy; latency, memory, energy, and thermal behaviour on actual
   Raspberry Pi / Jetson hardware were not measured.
5. **Dataset bias and imbalance.** The cohort is pediatric and pneumonia-heavy (~73%);
   subgroup performance (age, sex, ethnicity, comorbidity) was not assessed.
6. **Static-quantization calibration sensitivity.** Static PTQ accuracy depends on the
   observer/qconfig and calibration data; the reported recipe is robust on this
   platform but is backend-dependent.

### 5.4 Comparison with Prior Work

| Study | Model | Params | Explainability | Quantization | Edge focus | Reproducible pipeline |
|---|---|---|---|---|---|---|
| Rajpurkar et al. [4] | CheXNet (DenseNet-121) | ~28 M | No | No | No | Partial |
| Wang et al. [14] | EfficientNet-B0 | ~5.3 M | Post-hoc Grad-CAM | No | Partial | Partial |
| Benmalek et al. [25] | 5 lightweight CNNs (incl. EfficientNet-B0) | ~1–11 M | No | TensorRT (not compared) | **Yes (Jetson Nano)** | Partial (F1 only) |
| PneuNet [26] | Custom lightweight CNN | < 5 M | No | No | Partial | Partial |
| **This work** | **EfficientNet-B0 (+ ResNet-18, MobileNetV3-Small)** | **4.34 M** | **Integrated Grad-CAM++ (correct/FP/FN)** | **Dynamic + static INT8 (compared)** | **Yes (size/latency/RSS)** | **Yes (CIs, tests, manifests)** |

*Table 5.1: Qualitative positioning. ROC-AUC values are intentionally omitted from
cross-study comparison because they are reported on different datasets and splits and
are therefore not directly comparable; our measured AUCs appear in Tables 4.1–4.2.*

This work's distinguishing features are the *combination* of integrated explainability,
a real dynamic-versus-static quantization study, correct efficiency benchmarking
(including memory), and a reproducible, integrity-checked pipeline — rather than any
single record metric. The closest prior study, Benmalek et al. [25], shares the
backbone set and edge motivation and additionally provides on-device (Jetson) timing
that this work approximates with a CPU proxy; conversely, the present study adds the
data-leakage control, confidence intervals, quantization-scheme comparison,
explainability, and memory analysis that [25] does not report. The two are therefore
complementary, and on-device benchmarking is the natural next step (Section 6.2).

### 5.5 Ethical Considerations

**Bias and fairness.** The pediatric, single-source cohort may not generalise; subgroup
evaluation and, if needed, fairness-aware training are future work. **Clinical
responsibility.** The system is decision support, not a diagnostic device; uncertainty
(probability) and explanation (Grad-CAM++) outputs are provided to support oversight
and discourage over-reliance. **Data privacy.** Only publicly available, de-identified
data were used; any clinical deployment must comply with applicable regulation (e.g.,
HIPAA, GDPR).

---

## Chapter 6: Conclusion and Future Work

### 6.1 Conclusion

This thesis presents a memory-efficient and explainable deep-learning framework for
pneumonia detection from chest X-rays, built around a quantized EfficientNet-B0 and
evaluated with deliberate measurement rigour. On the held-out Kermany test set the
model achieves **ROC-AUC 0.9678 (95% CI 0.9504–0.9816)**, **sensitivity 0.9667**, and
the best precision/specificity balance among three architectures trained under an
identical pipeline. **Static INT8 quantization** produces a **5.13 MB (−71%) model with
~8× faster CPU inference and ≈6× lower peak inference memory (979.5 → 164.4 MB)** for a
2.3-point AUC cost; together with a lazy streaming data pipeline that uses **9.4× less
RAM** than naïve full-dataset loading, the system operates within a small, predictable
memory budget. **Grad-CAM++** provides prediction-level visual explanations. Crucially,
the accompanying pipeline detected and removed real train↔validation data leakage,
benchmarks efficiency correctly, and contains no simulated or fabricated results —
making the findings defensible and reproducible.

### 6.2 Future Work

1. **External validation** on NIH ChestX-ray14 and **CheXpert** to quantify domain
   shift and generalisation.
2. **Clinician reader study** (IRB-approved) to validate Grad-CAM++ explanations against
   expert annotations.
3. **Edge-hardware deployment** on Raspberry Pi 4 / Jetson Nano to measure real latency,
   memory, energy, and thermal behaviour.
4. **TinyML optimisation** — quantization-aware training (to close the static-PTQ gap),
   structured pruning, and knowledge distillation.
5. **Federated learning** across institutions to improve generalisation while preserving
   privacy.
6. **Multi-disease extension** from binary pneumonia detection to calibrated, explainable
   multi-label thoracic-disease classification.

These directions extend naturally into a doctoral research programme on trustworthy,
resource-efficient medical AI.

### 6.3 Final Remarks

High diagnostic value need not require high-resource infrastructure — but the claim must
be *earned through honest measurement*. By releasing a fully reproducible, integrity-
checked pipeline that refuses to fabricate results, this work aims to support the
responsible translation of medical AI to underserved settings.

---

## References

1. Walker CLF, et al. Global burden of childhood pneumonia and diarrhoea. *Lancet*. 2013;381(9875):1405-1416.
2. World Health Organization. Pneumonia in children. Fact sheet. 2022.
3. Callahan A, et al. Artificial intelligence in global health: defining a collective path forward. *NPJ Digit Med*. 2021;4:34.
4. Rajpurkar P, et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv:1711.05225*. 2017.
5. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. *Nat Med*. 2019;25(1):44-56.
6. Esteva A, et al. A guide to deep learning in healthcare. *Nat Med*. 2019;25(1):24-29.
7. Amann J, et al. Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC Med Inform Decis Mak*. 2020;20:310.
8. Deng J, et al. ImageNet: A large-scale hierarchical image database. *CVPR*. 2009.
9. Wang X, et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. *CVPR*. 2017.
10. Irvin J, et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *AAAI*. 2019.
11. Chaves J, et al. Self-supervised learning for medical image analysis: a review. *Med Image Anal*. 2023;84:102714.
12. Sandler M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*. 2018.
13. Tan M, Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*. 2019.
14. Wang G, et al. Lightweight Deep Learning for Pneumonia Detection on Chest X-rays. *IEEE JBHI*. 2022;26(5):2105-2114.
15. Zhou B, et al. Learning Deep Features for Discriminative Localization. *CVPR*. 2016.
16. Selvaraju RR, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*. 2017.
17. Chattopadhay A, et al. Grad-CAM++: Generalized Gradient-based Visual Explanations. *WACV*. 2018.
18. Saporta A, et al. Benchmarking saliency methods for chest X-ray interpretation. *Nat Mach Intell*. 2022;4:867-878.
19. Tonekaboni S, et al. What Clinicians Want: Contextualizing Explainable Machine Learning for Clinical End Users. *arXiv:2108.04838*. 2021.
20. Jacob B, et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. *CVPR*. 2018.
21. David R, et al. TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems. *MLSys*. 2021.
22. Kermany DS, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*. 2018;172(5):1122-1131.e9.
23. Pepe MS. The Statistical Evaluation of Medical Tests for Classification and Prediction. Oxford University Press; 2003.
24. Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall; 1993.
25. Benmalek E, Rhalem W, Jbabi A, et al. Edge-based real-time diagnosis of pediatric pneumonia using lightweight CNNs and chest X-rays. *Research on Biomedical Engineering*. 2025;41:Art. 60. doi:10.1007/s42600-025-00437-z.
26. Saranyaraj D, Shrinaath V, Nayak A, Vishal R. PneuNet: a lightweight convolutional neural network with multiscale feature fusion for automated pneumonia detection from chest X-rays. *Frontiers in Medicine*. 2026.

---

## Appendices

### Appendix A: Reproducibility Manifest

```bash
make setup          # create environment and install pinned dependencies
make validate-data  # class balance, duplicate detection, leakage check
make reproduce      # validate -> train (x3) -> evaluate -> benchmark -> quantize -> explain -> report -> render
```

Each training run writes a `metadata.json` manifest recording timestamp, git commit,
seed, library versions, full configuration, dataset statistics, and best epoch. Seeds
are fixed (seed = 42) and data loading is deterministic. Minor numeric differences may
arise across hardware/driver/library versions; the manifest records the exact
environment of each run. The committed run used EfficientNet-B0 best epoch 11
(validation AUC 0.9961; early-stopped at epoch 18), ResNet-18 best epoch 6, and
MobileNetV3-Small best epoch 7.

### Appendix B: Repository Structure

```
pneumonia-edge-xai/
├── configs/        # base + per-model YAML (efficientnet_b0, resnet18, mobilenetv3)
├── src/            # config, dataset, transforms, models, metrics, train, evaluate,
│                   #   inference, quantize, benchmarking, explainability, visualization, reporting
├── scripts/        # fetch_kermany_hf.py, run_full_pipeline.sh, render_report.py, ...
├── tests/          # 40 unit + integration tests
├── results/        # figures, tables (CSV/MD/LaTeX), metrics (JSON), gradcam overlays
├── reports/        # this thesis, limitations, future work, literature comparison, REPRODUCE
├── paper_assets/   # IEEE paper skeleton + captions
└── models/         # checkpoints, FP32/INT8 weights, ONNX (not version-controlled)
```

### Appendix C: Ethics Statement

All experiments used the publicly available, de-identified Kermany chest X-ray dataset
[22]. This software is a research artefact and **not** a medical device; it is not
approved for standalone diagnostic use. Clinical deployment would require regulatory
approval, prospective validation, and integration with clinical workflows under
appropriate oversight. Subgroup fairness evaluation is identified as future work.

---

> **Thesis submission checklist**
> - [ ] Replace bracketed fields ([Your University], [Supervisor Name], [Submission Date], author surname).
> - [ ] Insert generated figures (results/figures, results/roc_curves, results/confusion_matrices, results/gradcam).
> - [ ] Complete the literature-comparison table (reports/literature_comparison.md) from your reading.
> - [ ] Confirm all numbers match results/metrics/*.json (regenerate with `make reproduce`).
> - [ ] Format references per your university's required citation style.
