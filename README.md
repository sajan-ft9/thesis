# Memory-Efficient, Explainable, Quantized Pneumonia Detection from Chest X-Rays
### A reproducible, deployment-honest medical-AI pipeline (EfficientNet-B0 · Grad-CAM++ · INT8 · external validation)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 40 passing](https://img.shields.io/badge/tests-40%20passing-brightgreen.svg)](tests/)
[![Reproducible](https://img.shields.io/badge/research-reproducible-success.svg)](#reproducibility--how-to-verify-every-claim)
[![No fabricated results](https://img.shields.io/badge/results-measured%2C%20not%20fabricated-critical.svg)](#scientific-integrity)

A research codebase that detects pneumonia in chest X-rays with a **lightweight, quantized,
explainable** model — and, more importantly, **evaluates it the way it would actually be
deployed**: with data-integrity checks, confidence intervals, honest memory benchmarking,
and **zero-shot external validation on a second dataset**.

> **The contribution is the *methodology*, not a new classifier.** Lightweight CNNs already
> solve pneumonia classification; what is uncommon — and what this project provides — is an
> *integrated, integrity-checked, deployment- and memory-honest, reproducible* evaluation:
> (1) data-leakage detection, (2) reproducibility, (3) explainability, (4) a real
> dynamic-vs-static quantization study, and (5) correct efficiency/memory measurement.

This repository accompanies an MSc thesis ([`reports/thesis_final.md`](reports/thesis_final.md))
and is structured for replication and review. **Every number below is measured by the code
and machine-checked against the source data** (see [Verify every claim](#reproducibility--how-to-verify-every-claim)).

---

## Headline results (measured, reproducible)

**Diagnostic performance — Kermany held-out test set (n = 624):**

| Metric | EfficientNet-B0 | 95% CI |
|---|---|---|
| ROC-AUC | **0.9678** | 0.9504 – 0.9816 |
| Sensitivity | 0.9667 | — |
| Specificity | 0.8376 | — |
| F1 | 0.9366 | — |

**Fair architecture comparison (identical pipeline):** MobileNetV3-Small AUC 0.9743 (1.08 M params) · EfficientNet-B0 0.9678 (4.34 M, best precision/specificity balance) · ResNet-18 0.9594 (11.31 M).

**Efficiency & memory (the "memory-efficient" claim, measured — not `tracemalloc`):**

| | FP32 | INT8 static (PTQ) | Gain |
|---|---|---|---|
| Model size | 17.66 MB | **5.13 MB** | −71% |
| CPU latency / image | ~120 ms | **~15 ms** | ~8× |
| Peak inference RAM | 979.5 MB | **164.4 MB** | ~6× |
| Streaming vs naïve data load | 335.8 MB | 3,140.9 MB | **9.4× less RAM** |

**External validation (zero-shot, no tuning) — RSNA Pneumonia, n = 12,024 (adult; different source):**

| Metric | RSNA (external) | Kermany (internal) |
|---|---|---|
| ROC-AUC | **0.8892** (95% CI 0.8825 – 0.8954) | 0.9678 |
| Sensitivity | 0.9553 | 0.9667 |

A modest, *expected* drop under pediatric → adult domain shift — reported honestly, with no
attempt to optimise the external score.

**Data integrity:** an automatic check found and removed **26 duplicate images** that caused
train↔validation leakage; the official test set was kept canonical and verified disjoint.

---

## Scientific integrity

This project **does not fabricate results**. There are **no** simulated radiologist surveys,
**no** invented metrics, and **no** misleading `tracemalloc` memory figures. Every quantitative
claim is produced by the code from raw data and is machine-verifiable:

```bash
make verify-numbers      # asserts 21/21 headline numbers in the thesis == results/metrics/*.json
```

---

## Key contributions

1. **Data-integrity** — automatic duplicate + train/val/test leakage detection and removal.
2. **Reproducibility** — seeded, configuration-driven, tested pipeline; per-run manifests; one-command regeneration.
3. **Explainability** — integrated Grad-CAM++ for correct / false-positive / false-negative cases (no simulated trust scores).
4. **Quantization** — transparent dynamic vs. static INT8 study, incl. the practical finding that *per-channel* quantization is required to avoid EfficientNet-B0 collapse.
5. **Deployment efficiency** — correct size / latency / throughput / peak-memory benchmarking + zero-shot external validation.

---

## Repository structure

```
.
├── configs/            # base + per-model YAML (efficientnet_b0, resnet18, mobilenetv3)
├── src/                # typed, tested package — all logic lives here (no notebook-only code)
│   ├── config.py utils.py dataset.py transforms.py models.py metrics.py
│   ├── train.py evaluate.py inference.py quantize.py benchmarking.py
│   └── explainability.py visualization.py reporting.py memory_profile.py
├── scripts/            # fetch_kermany_hf.py, fetch_rsna_hf.py, build_rsna_subset.py,
│                       #   make_synthetic_data.py, run_smoke_test.sh, render_report.py,
│                       #   verify_thesis_numbers.py
├── tests/              # 40 unit + integration tests (CPU, offline, synthetic fixtures)
├── results/            # figures, tables (CSV/MD/LaTeX), metrics (JSON), Grad-CAM overlays
├── reports/            # thesis_final.md (the thesis), RESEARCH_PLAN.md, SUPERVISOR_BRIEF.md,
│                       #   related_work_annotated.md, limitations.md, future_work.md, REPRODUCE.md
├── paper_assets/       # IEEE conference-paper draft + captions
├── notebooks/          # thin demo that only calls into src/
└── Makefile  pyproject.toml  requirements.txt  environment.yml  CITATION.cff  LICENSE
```

---

## Installation

```bash
make setup                       # creates .venv and installs pinned dependencies
# or manually:
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Reference environment (verified): **Python 3.12, PyTorch 2.12, torchvision 0.27**, on
Apple-silicon (Metal/MPS) — **no CUDA required**. For an NVIDIA GPU, install the matching
`torch` build from [pytorch.org](https://pytorch.org) and keep the other pins.

---

## Quick start — prove the pipeline works in ~2 minutes (no dataset needed)

```bash
make smoke                       # end-to-end run on tiny SYNTHETIC data (not research data)
```
This exercises validate → train (3 models) → evaluate → benchmark → quantize → Grad-CAM++ →
report, confirming the whole pipeline executes.

---

## Reproducing the real results

### 1. Get the data
```bash
python scripts/fetch_kermany_hf.py          # Kermany pediatric CXR (credential-free HF mirror, ~1.2 GB)
make validate-data                          # class balance + duplicate + leakage report
```
See [`data/README.md`](data/README.md) for details and alternatives.

### 2. Full pipeline (train → evaluate → quantize → benchmark → explain → memory → report)
```bash
make reproduce
```
Or stage by stage: `make train train-baselines evaluate benchmark quantize explain memory report render`.
Every stage is also a CLI, e.g. `python -m src.train --config configs/efficientnet_b0.yaml`.

### 3. External validation (optional, ~3.5 GB download)
```bash
make external-rsna                          # fetch RSNA -> build balanced subset -> zero-shot inference
```

---

## Testing & verification

```bash
make test                # 40 unit + integration tests (offline, CPU, synthetic fixtures)
make verify-numbers      # every headline number in the thesis == results/metrics/*.json (21/21)
make lint                # ruff (style + import order)
```

### Reproducibility & how to verify every claim
Each claim in the thesis maps to a command — a reviewer can check them independently:

| Claim | Verify with |
|---|---|
| Test AUC 0.9678 (95% CI), sensitivity/specificity | `make train evaluate` → `results/metrics/efficientnet_b0_test_metrics.json` |
| 3-model comparison (identical pipeline) | `make train train-baselines evaluate` → `results/tables/table3_model_comparison.md` |
| INT8: −71% size, ~8× faster, ~6× less RAM | `make quantize benchmark memory` → `results/metrics/*_quantization.json`, `memory_profile.json` |
| Streaming uses 9.4× less RAM than naïve load | `make memory` → `results/metrics/memory_profile.json` |
| 26 duplicates / train-val leakage removed | `make validate-data` → `results/metrics/dataset_validation.json` |
| RSNA external AUC 0.8892 (zero-shot) | `make external-rsna` → `results/metrics/rsna_external_metrics.json` |
| Thesis numbers match the data exactly | `make verify-numbers` (21/21 PASS) |

Determinism: a single `seed_everything()` seeds Python/NumPy/PyTorch and makes DataLoaders
deterministic; every training run writes a `metadata.json` manifest (timestamp, git SHA, seed,
library versions, full config, dataset stats). Thesis numbers are injected from results files —
never hand-typed.

---

## Documentation

| Document | What it is |
|---|---|
| [`reports/thesis_final.md`](reports/thesis_final.md) | **The thesis** — full write-up with real, verified results |
| [`reports/REPRODUCE.md`](reports/REPRODUCE.md) | Step-by-step reproduction guide |
| [`reports/related_work_annotated.md`](reports/related_work_annotated.md) | Annotated bibliography (10 closely-related papers) |
| [`reports/RESEARCH_PLAN.md`](reports/RESEARCH_PLAN.md) | Roadmap to a conference paper / PhD directions |
| [`paper_assets/paper_ieee.tex`](paper_assets/paper_ieee.tex) | IEEE conference-paper draft (real numbers) |

---

## Limitations (stated plainly)

Single-institution **pediatric** training data; external validation is a **single, exploratory**
zero-shot probe (RSNA) — not multi-dataset, patient-level, or a clinical reader study; efficiency
is measured on CPU/GPU as an **edge proxy** (no physical Raspberry Pi / Jetson run);
explainability is **qualitative** (Grad-CAM++ not validated against expert annotations). See
[`reports/limitations.md`](reports/limitations.md). **This is a research artifact, not a medical
device, and is not for clinical use.**

---

## Citation

If you use this work, please cite it (see [`CITATION.cff`](CITATION.cff)) and the Kermany dataset:

> Kermany DS, et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
> Learning.* Cell. 2018;172(5):1122-1131.e9.

## Author

**Sajan** — MSc Information Technology · 📧 sajankhad2@gmail.com
*(institution / links to be added)*

## License

Code: **MIT** (see [`LICENSE`](LICENSE)). The Kermany and RSNA datasets retain their own licenses
(CC BY 4.0 and the RSNA challenge terms, respectively) and are not redistributed here.
