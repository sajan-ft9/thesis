# Annotated Bibliography — Core Reading for the Thesis

Ten closely-related papers, grouped by theme. Each entry gives an accurate citation,
a short summary, and **relevance / how this thesis differs**. Read these before the
supervisor meeting; they cover the problem, the lightweight/edge baselines we compare
against, the explainability tooling, and the quantization foundations.

> Verification note: citations below were checked against the publishers' pages.
> A few additional relevant papers (e.g., a *Sci. Rep.* 2025 attention-guided study and
> an MDPI Kermany-leakage study) were behind access walls at review time and are listed
> as "to verify" at the end rather than cited with unverified authors.

---

## A. Problem & dataset

**1. Kermany DS, Goldbaum M, Cai W, et al. "Identifying Medical Diagnoses and Treatable
Diseases by Image-Based Deep Learning." *Cell*, 172(5):1122–1131, 2018.**
The source of the pediatric CXR pneumonia dataset used here; introduces transfer-learning
diagnosis and occlusion-based explanations.
*Relevance:* our dataset and benchmark. *We differ:* patient-/content-level integrity
checking, confidence intervals, and a deployment-focused efficiency study they do not perform.

**2. Rajpurkar P, Irvin J, Zhu K, et al. "CheXNet: Radiologist-Level Pneumonia Detection
on Chest X-Rays with Deep Learning." *arXiv:1711.05225*, 2017.**
A 121-layer DenseNet for CXR classification; the canonical high-capacity baseline.
*Relevance:* motivates the need for lighter models. *We differ:* ~6× fewer parameters,
INT8 quantization, and explicit memory/latency budgets.

## B. Lightweight & edge CXR pneumonia (closest matches)

**3. Benmalek E, Rhalem W, Jbabi A, et al. "Edge-based real-time diagnosis of pediatric
pneumonia using lightweight CNNs and chest X-rays." *Research on Biomedical Engineering*,
41:Art. 60, 2025. doi:10.1007/s42600-025-00437-z.**
Benchmarks five lightweight CNNs (ResNet-18, MobileNet, ShuffleNet, SqueezeNet,
EfficientNet-B0) on the same dataset and deploys the best on an NVIDIA Jetson Nano via
TensorRT (F1 95.7/95.1/94.4% for MobileNet/EfficientNet-B0/ResNet-18).
*Relevance:* **the single closest prior work** — same backbones, same data, edge focus.
*We differ:* data-leakage control, bootstrap CIs, a dynamic-vs-static INT8 comparison,
memory-footprint analysis, and integrated Grad-CAM++ — none of which they report (they
report F1 only). *They have:* real on-device (Jetson) timing, which we approximate on CPU.

**4. Saranyaraj D, Shrinaath V, Nayak A, Vishal R. "PneuNet: a lightweight convolutional
neural network with multiscale feature fusion for automated pneumonia detection from
chest X-rays." *Frontiers in Medicine*, 2026.**
A custom efficient CNN (depthwise-separable convolutions, SE blocks, ASPP, learnable
pooling) reaching competitive accuracy with few parameters.
*Relevance:* shows custom lightweight designs are viable. *We differ:* we use a standard,
reproducible backbone + quantization + explainability rather than a bespoke architecture.

**5. Nettur SB, Karpurapu S, Nettur U, et al. "Lightweight Weighted Average Ensemble Model
for Pneumonia Detection in Chest X-Ray Images." *arXiv* (2501.16249), 2025.**
A MobileNetV2 + NASNetMobile weighted ensemble reporting 98.63% accuracy on pediatric CXR.
*Relevance:* the high-accuracy efficient end of the field. *We differ:* a single quantized
model (no ensemble cost), with honest evaluation and a memory budget; we do **not** chase
peak accuracy.

**6. Tan M, Le Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural
Networks." *ICML*, 2019.**
Compound scaling of depth/width/resolution; EfficientNet-B0 is the small, efficient base.
*Relevance:* our backbone. *We use it as-is* and study its quantization behaviour.

## C. Explainability

**7. Selvaraju RR, Cogswell M, Das A, et al. "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization." *ICCV*, 2017.**
Gradient-weighted class-activation maps for CNN explanations.
*Relevance:* foundation of our XAI. *We use* its successor (Grad-CAM++).

**8. Chattopadhay A, Sarkar A, Howlader P, Balasubramanian VN. "Grad-CAM++: Generalized
Gradient-based Visual Explanations for Deep Convolutional Networks." *WACV*, 2018.**
Improves localization for multiple and small objects over Grad-CAM.
*Relevance:* our explanation method, applied to correct/FP/FN cases.

**9. Saporta A, Gui X, Agrawal A, et al. "Benchmarking saliency methods for chest X-ray
interpretation." *Nature Machine Intelligence*, 4:867–878, 2022.**
Shows saliency maps often align poorly with expert localization — they must be validated,
not assumed.
*Relevance:* **why we make no radiologist-alignment claim** and treat Grad-CAM++ as
qualitative; motivates the future quantitative-localization work.

## D. Quantization & efficient inference

**10. Jacob B, Kligys S, Chen B, et al. "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference." *CVPR*, 2018.**
Foundations of INT8 quantization (per-channel weights, calibration) for fast integer
inference.
*Relevance:* underpins our PTQ; explains why **per-channel** quantization matters — exactly
the detail that keeps EfficientNet-B0 from collapsing in our static-PTQ experiment.

---

## To verify and add (access-walled at review time)
- A *Scientific Reports* (2025) attention-guided "trustworthy pneumonia detection" study
  (s41598-025-23664-x) — relevant to the trust/XAI framing.
- An MDPI *Applied Sciences* (2025) study (PELM, 15/12/6487) that **re-partitions Kermany
  at the patient level to remove leakage** — directly supports our data-integrity point;
  worth citing once you confirm authors.
- "Explanatory Analysis and Rectification of the Pitfalls in COVID-19 Datasets"
  (arXiv:2111.05679) — broader data-integrity/leakage context.

## How to use this for the meeting
Be ready to say, in one or two sentences each, what papers 3, 5, and 9 do and how your
work differs — those are the ones a supervisor will probe (the closest competitor, the
high-accuracy ensemble, and the saliency-validation caveat).
