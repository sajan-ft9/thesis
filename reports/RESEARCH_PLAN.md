# Master Plan — From Working Thesis to a Publishable Paper and a PhD-RA-Ready Portfolio

**Author:** Sajan · **Date:** 2026-06-14 · **Status:** strategy (literature-grounded)

This plan turns the current, honest, reproducible pipeline into (a) a stronger
Master's thesis, (b) a paper realistically acceptable at an IEEE venue, and (c) a
portfolio that signals "ready to do research" to a PhD advisor. It is grounded in a
scan of recent (2022–2026) literature (see §8 Reading List).

---

## 1. Honest starting point

What you have is a genuine asset: a typed, tested, reproducible pipeline; three
models trained under one protocol; real test metrics with CIs; dynamic+static INT8;
Grad-CAM++; correct efficiency benchmarking; and a real data-integrity finding
(byte-duplicate removal). **What you do not yet have is a novel research
contribution.** Pneumonia + Kermany + EfficientNet + Grad-CAM + INT8 is one of the
most saturated topics in medical-imaging ML; raw accuracy (~0.97 AUC) is not
competitive (ensembles/ViT hybrids report 0.97–0.99), and "lightweight CNN + Jetson
edge deployment" is already published (e.g., Imran et al. 2025, *Res. Biomed. Eng.*).

**Conclusion:** do not compete on accuracy or on "a framework." Compete on
**rigour, honesty, and a measurable question nobody answers cleanly.**

---

## 2. The spine: one sharp, defensible contribution

> **Thesis/paper statement —** *"When you evaluate pneumonia CXR models the way they
> would actually be deployed — with patient-level splits, INT8 quantization, and
> cross-dataset testing — how much do reported gains shrink, and does compression
> degrade not just accuracy but the **faithfulness of the explanations**?"*

This reframes the project from "build a detector" to **"a deployment-honest,
explainability-aware benchmark of efficient pneumonia detection."** It is novel in
*combination* and, critically, every claim is *measurable*. The four pillars:

1. **Honest evaluation** — patient-level splitting (fix the known Kermany leakage) and
   quantify the optimism vs the naïve official split.
2. **Cross-dataset generalization** — train on Kermany, test on **RSNA/NIH** (adult,
   multi-source); report the generalization gap.
3. **Quantization done properly** — PTQ (per-channel) + **QAT**, reporting the full
   accuracy/size/latency frontier.
4. **Quantitative, quantization-aware explainability (the novel hook)** — evaluate
   Grad-CAM++ localization against **RSNA bounding boxes** (IoU / pointing-game), and
   measure **how INT8 quantization changes explanation fidelity** — a question the
   literature largely ignores (papers check accuracy after quantization, not
   explanations).

Pillar 4 is the differentiator: *"Does a 70.5%-smaller INT8 model still look at the
right place?"* is a clean, publishable research question that ties efficiency and
explainability together.

---

## 3. Workstreams (what to build, why it adds value, feasibility)

Feasibility tags: 🟢 doable now on your Mac (MPS) · 🟡 doable but heavier (downloads/compute) · 🔴 needs extra hardware.

### W1 — Patient-level splitting & leakage quantification 🟢
- **Why:** Kermany filenames encode patient IDs (`person{N}_{bacteria|virus}_*`).
  Same patient appears across the official train/test → optimistic metrics. Most
  papers ignore this; doing it right is concrete rigor.
- **Tasks:** re-fetch Kermany *with original filenames* (Kaggle/Mendeley, not the
  re-encoded HF mirror); parse patient IDs; add `GroupShuffleSplit`/patient-level
  split to `src/dataset.py`; report metrics under (i) official split, (ii) random
  patient-level split, (iii) byte-dedup (current). Quantify the AUC/accuracy delta.
- **Deliverable:** a table "evaluation protocol vs measured performance" — likely
  your single most reviewer-pleasing result.

### W2 — Cross-dataset external validation 🟡
- **Why:** single-dataset evaluation is the #1 rejection reason in medical-ML.
- **Tasks:** ingest **RSNA Pneumonia Detection Challenge** (NIH-derived, adult, with
  boxes) and/or NIH ChestX-ray14 pneumonia subset; build an adapter to the existing
  `Dataset`; evaluate Kermany-trained models zero-shot, then with light fine-tuning;
  report the domain-shift gap honestly.
- **Deliverable:** generalization table + ROC overlay (in-distribution vs external).

### W3 — Quantitative explainability vs ground truth 🟡 (novel hook)
- **Why:** turns Grad-CAM++ from decoration into measurement; RSNA gives boxes.
- **Tasks:** implement localization metrics in `src/explainability.py` —
  **IoU** of thresholded CAM vs box, **pointing game** (does the CAM peak fall in a
  box?), and **energy-in-box / Lung-Attention-Ratio**. Evaluate per model.
- **Deliverable:** "explanation localization" table + qualitative panels (TP/FP/FN).

### W4 — Quantization frontier: PTQ → QAT, accuracy *and* explanation fidelity 🟢/🟡
- **Why:** PTQ alone loses 2.3 AUC pts; QAT should recover it — a satisfying arc; and
  the explanation-fidelity-under-quantization analysis is the novel question.
- **Tasks:** add **QAT** (`torch.ao.quantization` FX QAT) to `src/quantize.py`;
  produce the frontier {FP32, dynamic, static-PTQ, QAT} × {size, latency, AUC,
  ECE-calibration, localization-IoU}. Add **explanation-fidelity metrics**: CAM
  agreement between FP32 and quantized models (cosine similarity / IoU of CAMs;
  rank correlation), so you can state "INT8 preserves/d egrades explanations by X."
- **Deliverable:** the headline figure — a single plot trading off size/latency vs
  AUC vs explanation-IoU across precisions.

### W5 — Statistical & calibration rigor 🟢
- **Why:** elevates from "good numbers" to "defensible numbers."
- **Tasks:** **DeLong test** for AUC differences between models/precisions;
  **Expected Calibration Error (ECE)** + reliability diagrams; report operating
  points (default vs Youden) consistently; McNemar test for paired error differences.
- **Deliverable:** significance-annotated comparison tables.

### W6 — Real edge-hardware benchmarking 🔴 (optional but high-credibility)
- **Why:** your title says "resource-constrained"; on-device numbers make it real.
- **Tasks:** if you can borrow a **Raspberry Pi 4 / Jetson Nano**, run ONNX/TFLite
  INT8 and log latency, memory, and energy (a USB power meter). If not, **state CPU
  results as an explicit proxy** (already done) and keep this as future work.
- **Deliverable:** on-device latency/energy table — or an honest "proxy + future work."

### W7 — Engineering/repo polish for PhD-RA signal 🟢
- CI (GitHub Actions running `pytest`), coverage badge, a 1-figure README result
  teaser, a short **arXiv/tech-report preprint**, a model card, and a DOI via Zenodo.
- A clean issue/PR history and a `CHANGELOG`. These signal "can ship research code."

---

## 4. Phased timeline (~8–10 focused weeks)

| Phase | Weeks | Workstreams | Output |
|---|---|---|---|
| P1 Honest eval | 1–2 | W1, W5 | Patient-level results + leakage-inflation table; stats |
| P2 Generalization | 3–4 | W2 | External (RSNA/NIH) validation table |
| P3 Quant frontier | 4–6 | W4 | PTQ→QAT frontier; calibration |
| P4 XAI measurement | 5–7 | W3, W4 | Localization-IoU + quantization-fidelity (the hook) |
| P5 Write-up | 7–9 | — | Update thesis Ch.3–5; draft IEEE paper |
| P6 Polish/submit | 9–10 | W6?, W7 | arXiv preprint, repo CI, conference submission |

Phases overlap; P1+P4 alone already make a publishable short paper.

---

## 5. Venue targeting (realistic, tiered)

- **After P1–P4 (recommended target):** **IEEE Access** (broad, rigorous, open
  access) or **IEEE J-BHI** (if cross-dataset + QAT + quantitative XAI all land);
  conferences **IEEE EMBC**, **IEEE ISBI**, **IEEE BHI**, **IEEE BIBM**.
- **Short/workshop now (with P1 + W3):** an ISBI/EMBC short paper or a MICCAI/NeurIPS
  workshop on trustworthy/efficient medical AI.
- **Not realistic regardless:** IEEE TMI / MICCAI main track (novelty bar too high
  for this scope) — don't aim there; it wastes a review cycle.

**Framing tip:** title the paper around the *question*, e.g. *"Deployment-Honest
Pneumonia Detection: Patient-Level Evaluation and Explainability-Preserving INT8
Quantization across Datasets."* Avoid "a novel framework for…".

---

## 6. Paper outline (IEEE, ~6–8 pages) built on the spine

1. **Intro** — deployment gap; the three honesty failures (leakage, single-dataset,
   accuracy-only-quantization); contributions = the four pillars.
2. **Related work** — lightweight CXR, XAI in CXR, quantization (cite §8); the gap:
   nobody jointly measures honest eval + quantization + *explanation fidelity*.
3. **Methods** — models; patient-level protocol; PTQ/QAT; Grad-CAM++ localization
   metrics; statistical tests.
4. **Experiments** — datasets (Kermany + RSNA/NIH).
5. **Results** — (a) protocol-vs-performance, (b) generalization, (c) quant frontier,
   (d) localization + quantization-fidelity (headline figure), (e) calibration/stats.
6. **Discussion/limitations** — pediatric source, no reader study, CPU proxy.
7. **Conclusion + future work** (→ your PhD directions).

---

## 7. PhD research-assistant positioning

Advisors hire RAs who can (1) frame a question, (2) run rigorous experiments, (3)
ship reproducible code, (4) write. This plan demonstrates all four. To maximise signal:
- **Lead with rigor, not accuracy.** Your differentiator is "I found and fixed the
  leakage everyone ignores; I measured whether compression breaks explanations." That
  is *research maturity* — exactly what advisors want.
- **One focused contribution > a kitchen-sink framework.** Pick pillar 4 as the story.
- **Public artifacts:** GitHub (CI + tests + README teaser), an arXiv preprint, a
  short blog/thread summarising the explanation-fidelity finding, a model card.
- **A 1-page research statement** derived from Ch.6 future work (external validation,
  QAT, federated, edge TinyML) — shows you can scope a multi-year program.
- **Email advisors** with: the preprint link, the repo, and a 3-sentence pitch of the
  explanation-fidelity-under-quantization result.

---

## 8. Reading list (read these, then 5–8 more they cite)

Efficiency / lightweight CXR (position your baselines):
- Imran et al. (2025), *Edge-based real-time diagnosis of pediatric pneumonia using
  lightweight CNNs*, Res. Biomed. Eng. — https://link.springer.com/article/10.1007/s42600-025-00437-z
- *PneuNet: lightweight multiscale CNN for pneumonia* — https://pmc.ncbi.nlm.nih.gov/articles/PMC12832688/
- *Lightweight Weighted Average Ensemble for Pneumonia* — https://arxiv.org/pdf/2501.16249

Data integrity / leakage (motivates W1):
- *Explanatory Analysis and Rectification of the Pitfalls in COVID-19 Datasets* — https://arxiv.org/pdf/2111.05679
- PELM (re-partitions Kermany to fix leakage) — https://www.mdpi.com/2076-3417/15/12/6487

Localization / XAI evaluation (motivates W3):
- *Weakly Supervised Pneumonia Localization … Grad-CAM* — https://arxiv.org/html/2511.00456
- RSNA challenge overview (boxes, IoU) — https://ajronline.org/doi/10.2214/AJR.19.21512
- Saporta et al. (2022), *Benchmarking saliency methods for CXR*, Nat. Mach. Intell.

Quantization (motivates W4):
- *Post-Training Quantization for 3D Medical Image Segmentation* — https://arxiv.org/html/2501.17343v1
- Jacob et al. (2018), integer-arithmetic inference (PTQ foundations).
- NVIDIA QAT + TAO toolkit (practical QAT) — https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/

Trustworthy/attention CXR (framing):
- *Trustworthy pneumonia detection … attention-guided DL* — https://www.nature.com/articles/s41598-025-23664-x

---

## 9. Risks & mitigations
- **RSNA/NIH download size/compute** (🟡): start with a subset; cache; run eval on CPU/MPS.
- **QAT instability on MPS** (🟡): QAT may need CPU/CUDA; fall back to CPU QAT or a
  cloud GPU session for that one experiment; document the environment.
- **No edge hardware** (🔴): keep CPU-as-proxy + future work; don't block the paper on it.
- **Novelty still questioned:** the explanation-fidelity-under-quantization result is
  the insurance — keep it central.

## 10. Immediate next actions (highest leverage first)
1. **W1**: re-fetch Kermany with original filenames; add patient-level split; produce
   the protocol-vs-performance table. *(biggest credibility gain, ~days)*
2. **W3 scaffold**: implement CAM localization metrics (works on RSNA later).
3. **W4**: add QAT + the FP32-vs-INT8 explanation-agreement metric.
4. **W2**: ingest RSNA for external eval + box-based localization.
5. **W5/W7**: DeLong/ECE + CI/preprint.
