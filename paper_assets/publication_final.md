# Memory-Efficient and Explainable Pneumonia Detection from Pediatric Chest X-rays: A Reproducible Evaluation of INT8 Post-Training Quantization

**Author:** Sajan Khadka  
**Affiliation:** [Department, University, Nepal]  
**Corresponding author:** sajankhad2@gmail.com  
**Manuscript status:** submission-ready technical draft; replace bracketed author metadata and apply the target journal template before submission.

## Abstract

### Background

Pneumonia classification from chest radiographs may support screening where specialist interpretation and computing resources are limited. A useful model must therefore be evaluated for discrimination, data integrity, explanation behavior, and resource cost rather than accuracy alone.

### Methods

We performed a reproducible binary classification study using the real Kermany pediatric chest-X-ray dataset. The official test set was held out, and a stratified validation split was created from the training pool. An automated byte-level and content-hash audit removed duplicate files before splitting and checked cross-split leakage. EfficientNet-B0 was compared with ResNet-18 and MobileNetV3-Small under the same training and evaluation protocol. The primary test threshold was locked at 0.5; a Youden threshold was computed from validation predictions only and treated as exploratory. We report bootstrap 95% confidence intervals, paired model comparisons, calibration statistics, qualitative Grad-CAM++ error analysis, on-disk artifact sizes, repeated CPU latency, and fresh-process resident-set-size measurements. Dynamic INT8 and static post-training INT8 quantization were evaluated using the QNNPACK backend.

### Results

The dataset audit removed 26 byte-identical duplicate files and left a leakage-clean split of 4,152 training, 1,038 validation, and 624 test images. On the held-out test set, EfficientNet-B0 achieved ROC-AUC 0.9678 (95% CI 0.9504–0.9816), accuracy 0.9183, sensitivity 0.9667, specificity 0.8376, and F1 0.9366. MobileNetV3-Small had the highest AUC (0.9743), but EfficientNet-B0 produced fewer false positives at the locked threshold (38 versus 72 for ResNet-18 and 77 for MobileNetV3-Small). Static INT8 reduced the measured on-disk artifact from 17.67 MB to 5.22 MB (70.5%) and isolated inference RSS from 559.7 MB to 222.7 MB, but reduced AUC to 0.9427 and increased latency from 174.7 ms to 233.0 ms per image in this CPU/QNNPACK run. Dynamic INT8 reduced artifact size by 5.5%, produced AUC 0.9683, and measured 136.9 ms latency. Streaming the training data used 134.7 MB RSS versus 3,141.1 MB for naïve full-tensor loading. A zero-shot probe on the real processed RSNA set (n=12,024) achieved AUC 0.8892 (95% CI 0.8825–0.8954), sensitivity 0.9553, specificity 0.6060, and F1 0.8132.

### Conclusions

EfficientNet-B0 provides strong internal discrimination and the best locked-threshold specificity/F1 balance among the evaluated models, but the experiments do not establish clinical utility or universal edge-device performance. Static INT8 is a storage and memory trade-off, not an unconditional speed improvement. The study’s main contribution is an auditable workflow that keeps real-data provenance, leakage control, threshold discipline, calibration, and resource measurements connected to every reported claim.

**Keywords:** pneumonia; chest radiography; pediatric imaging; EfficientNet; explainable AI; Grad-CAM++; post-training quantization; INT8; reproducible medical AI; edge inference.

## 1. Introduction

Pneumonia remains an important cause of childhood morbidity and mortality, particularly in settings where radiology capacity is limited. Deep neural networks can assist chest-radiograph screening, but a high test score alone does not establish a clinically useful system. Public chest-X-ray datasets can contain duplicate or related images, performance can vary under dataset shift, operating thresholds can be selected incorrectly, and resource claims can be overstated when model size or memory is measured indirectly.

This study narrows the question to a defensible engineering and evaluation target: can a compact pediatric pneumonia classifier preserve useful discrimination while reducing storage and inference memory, and can that claim be reproduced from exact real data? The work compares three compact or moderately compact convolutional backbones, evaluates a primary EfficientNet-B0 model with confidence intervals and calibration metrics, examines Grad-CAM++ outputs qualitatively, and measures dynamic and static INT8 post-training quantization.

The contributions are:

1. a real-data, leakage-audited Kermany evaluation with a locked test threshold;
2. an identical-pipeline comparison of EfficientNet-B0, ResNet-18, and MobileNetV3-Small;
3. a quantization study that reports artifact size, accuracy, AUC, latency, backend, and observer configuration together;
4. isolated process-RSS and streaming-versus-naïve memory measurements; and
5. an exploratory zero-shot RSNA probe and qualitative explanation/error analysis, explicitly separated from clinical validation.

## 2. Related work

Kermany et al. demonstrated image-based diagnosis on pediatric chest radiographs and made the dataset widely used for pneumonia classification. CheXNet established the importance of convolutional radiograph classification at scale. EfficientNet introduced compound scaling for accuracy-efficient CNN design. Grad-CAM++ provides class-discriminative visual explanations, but saliency maps are not evidence of clinical correctness; in this study they are used only for qualitative error inspection. Quantization work by Jacob et al. motivates integer inference, while current reporting guidance such as CLAIM and TRIPOD+AI emphasizes transparent dataset, model, evaluation, and limitation reporting. Prior dataset-shift studies also show that strong internal performance may not transfer across hospitals or acquisition protocols.

## 3. Materials and methods

### 3.1 Dataset and integrity audit

The primary dataset was the real Kermany pediatric chest-X-ray corpus already present in `data/raw/chest_xray`. Its official test set contains 624 images. The training pool was split into training and validation partitions with the project’s seeded stratified procedure. The final counts were 4,152 training images, 1,038 validation images, and 624 test images; the combined primary corpus contains 5,814 images after the project’s deduplication policy.

The integrity audit checked image bytes and content hashes, duplicate groups, path-based split membership, and cross-split content overlap. Twenty-six byte-identical duplicate files were removed from the training pool. The resulting validation report records the final class distributions and reports the split as leakage-clean under the implemented checks. The audit does not prove patient-level independence because the source release does not provide sufficient patient identifiers for a complete independent-patient audit.

### 3.2 Models and training

The primary model was EfficientNet-B0 with a binary classification head. ResNet-18 and MobileNetV3-Small were trained through the same project pipeline as baselines. Training used the committed YAML configurations, seeded initialization, staged transfer learning, AdamW optimization, learning-rate scheduling, label smoothing, early stopping on validation AUC, and deterministic split construction where supported by the runtime. The manuscript reports the existing trained checkpoints; no new synthetic images, labels, or synthetic performance values are used.

### 3.3 Evaluation and threshold policy

The held-out test set was evaluated once using the locked configuration threshold of 0.5. Accuracy, precision, sensitivity, specificity, F1, and ROC-AUC were calculated. Accuracy and AUC confidence intervals use 1,000 bootstrap resamples with the committed seed and alpha. Validation predictions were collected separately, and the validation-only Youden diagnostic selected threshold 0.1516. That threshold was not used to tune or report the primary test estimate; operating-point selection for deployment remains a separate validation and clinical-policy decision.

Model comparisons use paired bootstrap AUC differences, DeLong tests, and exact McNemar tests on paired errors. The reported p-values are exploratory pairwise results and are not presented as a confirmatory multiplicity-adjusted clinical claim. Calibration was summarized using ECE, Brier score, and logistic calibration slope/intercept.

### 3.4 Quantization and efficiency measurement

Dynamic INT8 quantization was applied to supported linear layers. Static PTQ used FX graph-mode quantization with per-channel symmetric INT8 weights, histogram activation observers, clean validation calibration data, and the QNNPACK backend. The static artifact was saved as a TorchScript `.pt` file; FP32 and dynamic artifacts were saved as state-dict files. Artifact size is the actual file size on disk, not a `BytesIO` estimate.

Latency was measured after warm-up with repeated CPU inference. Runtime memory was measured in separate fresh subprocesses using sampled process RSS during the full test inference pass, with `num_workers=0` for comparability. The streaming-memory experiment compares one-pass lazy loading with a deliberately naïve full-dataset float32 tensor allocation; it is a loader-design experiment, not a claim about a clinical device’s total operating system memory.

### 3.5 Explainability and external probe

Grad-CAM++ overlays were generated for true positives, true negatives, false positives, and false negatives. They are reported as qualitative visual error analysis and were not scored against radiologist annotations. An exploratory RSNA probe used the real processed dataset in `data/processed/rsna_external`, with the project’s documented label mapping and no threshold optimization on that external set. It is evidence of domain-shift behavior, not clinical external validation.

## 4. Results

### 4.1 Dataset and model performance

The audit found 26 byte-identical duplicate files and produced the final split of 4,152/1,038/624 images for train/validation/test. The test set contained 234 normal and 390 pneumonia images.

| Model | Parameters (M) | AUC | Accuracy | Sensitivity | Specificity | F1 | False positives / false negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 4.34 | 0.9678 | 0.9183 | 0.9667 | **0.8376** | **0.9366** | **38 / 13** |
| ResNet-18 | 11.31 | 0.9594 | 0.8798 | 0.9923 | 0.6923 | 0.9117 | 72 / 3 |
| MobileNetV3-Small | 1.08 | **0.9743** | 0.8750 | 0.9974 | 0.6709 | 0.9089 | 77 / 1 |

EfficientNet-B0 was not the highest-AUC model, but it had the best locked-threshold specificity and F1 and substantially fewer false positives. The AUC difference between EfficientNet-B0 and MobileNetV3-Small was not statistically distinguishable in the paired tests; therefore, the selection is a multi-objective engineering choice rather than a claim of superior discrimination.

EfficientNet-B0’s extended test statistics were MCC 0.8250, balanced accuracy 0.9021, ECE 0.0354, and Brier score 0.0633. ResNet-18 and MobileNetV3-Small had ECE values of 0.0529 and 0.0559, respectively.

### 4.2 Quantization trade-offs

| Variant | Artifact (MB) | Size reduction | Accuracy | AUC | AUC change | Latency (ms/image) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 17.667 | — | 0.9183 | 0.9678 | — | 174.684 ± 22.594 |
| INT8 dynamic | 16.688 | 5.5% | 0.9199 | 0.9683 | −0.05% | 136.913 ± 19.035 |
| INT8 static PTQ | **5.219** | **70.5%** | 0.7981 | 0.9427 | −2.51% | 233.028 ± 17.047 |

Static PTQ provides the strongest storage reduction and lower isolated RSS, but the measured backend did not accelerate it. At the locked threshold, static PTQ increased sensitivity to 0.9949 while specificity fell to 0.4701. This is a deployment trade-off: a storage-constrained screening system might value the artifact reduction, but it requires a separately validated operating-point policy and cannot be described as an unconditional improvement.

### 4.3 Memory and CPU measurements

| Measurement | FP32 | INT8 dynamic | INT8 static PTQ |
|---|---:|---:|---:|
| Isolated inference RSS delta (MB) | 559.7 | 541.1 | **222.7** |
| Artifact size (MB) | 17.67 | 16.69 | **5.22** |

The streaming loader peaked at 134.7 MB RSS, compared with 3,141.1 MB for naïve full-dataset float32 loading, a 23.3-fold difference in this controlled experiment. The primary EfficientNet FP32 benchmark measured 164.05 ms mean CPU latency and 27.0 images/s throughput; baseline means were 103.53 ms for ResNet-18 and 76.81 ms for MobileNetV3-Small. These are containerized CPU proxy measurements, not physical edge-device results.

### 4.4 External domain-shift probe

On the real processed RSNA probe (n=12,024; 6,012 examples per mapped class), EfficientNet-B0 achieved AUC 0.8892 (95% CI 0.8825–0.8954), sensitivity 0.9553, specificity 0.6060, and F1 0.8132 at the locked threshold. The lower AUC and specificity relative to Kermany are consistent with domain shift, but no causal explanation or clinical generalization claim is made.

### 4.5 Qualitative explanation and errors

The generated Grad-CAM++ records include correct classifications, false positives, and false negatives. Correct positive cases generally highlight pulmonary regions, while errors include subtle or ambiguous cases and spurious regions. Because no radiologist localization labels were used, the maps demonstrate inspectability only; they do not prove anatomical faithfulness or clinical trustworthiness.

## 5. Discussion

The principal result is not that one backbone is universally best. MobileNetV3-Small achieved the highest AUC and smallest FP32 artifact among the three, while EfficientNet-B0 achieved the best locked-threshold specificity/F1 balance. The appropriate choice depends on whether the deployment objective prioritizes ranking discrimination, false-alarm burden, storage, or sensitivity.

The quantization result is similarly conditional. Dynamic INT8 preserved the primary discrimination and improved measured latency modestly but saved little storage. Static INT8 reduced the artifact by 70.5% and the isolated RSS delta by approximately 60%, yet was slower and less accurate at the locked threshold in the tested QNNPACK environment. This demonstrates why “INT8 is faster” should not be reported without the backend, operator coverage, calibration data, timing protocol, and artifact provenance.

The external RSNA probe reinforces the need for dataset-shift evaluation: discrimination remained useful but specificity decreased materially. The probe is not a clinical validation study because the label mapping, dataset composition, and patient-level metadata are not equivalent to a prospective multi-centre cohort.

## 6. Limitations

1. The primary evaluation is based on a single pediatric source cohort and may not transfer across institutions, scanners, age groups, or disease prevalence.
2. Patient-level independence cannot be fully verified from the available source metadata; the implemented byte/content audit is not a substitute for patient identifiers.
3. The RSNA experiment is one exploratory zero-shot probe, not multi-centre external validation.
4. The project reports existing trained checkpoints and one training seed for the headline model comparison; multi-seed uncertainty should be added before a confirmatory claim.
5. Grad-CAM++ is qualitative and has no radiologist or bounding-box faithfulness assessment.
6. CPU/QNNPACK timings are platform-specific; no Raspberry Pi, Jetson, hospital workstation, or prospective clinical workflow was tested.
7. The static PTQ experiment uses one calibration configuration. Quantization-aware training, alternative backends, and a separately selected deployment threshold remain future work.

## 7. Reproducibility and data availability

All reported values are generated from the local real datasets, checkpoints, configuration files, and scripts in this repository. The Docker path is:

```bash
docker compose run --build --rm real-validate
docker compose run --rm real-validate python -m src.evaluate --checkpoint /app/models/efficientnet_b0_best.pth --device cpu
docker compose run --rm real-validate python -m src.quantize --checkpoint /app/models/efficientnet_b0_best.pth
docker compose run --rm real-validate python -m src.memory_profile --config configs/efficientnet_b0.yaml --checkpoint /app/models/efficientnet_b0_best.pth
docker compose run --rm real-validate python scripts/verify_thesis_numbers.py
```

The repository does not embed the medical images. Users must obtain the Kermany and RSNA data through their official sources and place them in the documented local paths. Dataset licenses and access conditions remain the responsibility of the user. Generated metrics are in `results/metrics/`, figures in `results/figures/`, and the developer/reader/audit documentation is in `reports/`.

## 8. Declarations

**Ethics approval:** This secondary analysis uses publicly released, de-identified datasets. Confirm the target journal’s requirements and add the applicable institutional statement before submission.  
**Competing interests:** [Complete before submission.]  
**Funding:** [Complete before submission.]  
**Author contributions:** [Complete before submission.]  
**Code and materials:** The code, configuration, Docker recipe, tests, and result-generation scripts are included in this project.  
**Clinical use:** The model is a research screening aid and is not approved for autonomous diagnosis.

## References

1. Kermany DS et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*. 2018;172(5):1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010
2. Rajpurkar P et al. CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225, 2017.
3. Tan M, Le Q. EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 2019.
4. Chattopadhay A et al. Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks. *WACV*, 2018. https://arxiv.org/abs/1710.11063
5. Jacob B et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html
6. Oakden-Rayner L et al. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. *Proceedings of Machine Learning Research*, 2020.
7. Zech JR et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLoS Medicine*. 2018;15(11):e1002683. https://doi.org/10.1371/journal.pmed.1002683
8. Mongan J et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 update. *Radiology: Artificial Intelligence*. 2024. https://doi.org/10.1148/ryai.240300
9. Collins GS et al. TRIPOD+AI statement. *BMJ*. 2024;385:e078378. https://www.bmj.com/content/385/bmj-2023-078378
