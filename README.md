# Memory-Efficient Explainable Pneumonia Detection
### Quantized EfficientNet-B0 for Resource-Constrained Healthcare Environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Reproducible](https://img.shields.io/badge/research-reproducible-success.svg)](#reproducibility)

A reproducible research framework for **explainable, resource-efficient pneumonia
detection** from chest X-rays. It pairs a lightweight **EfficientNet-B0** backbone
(benchmarked against **ResNet-18** and **MobileNetV3-Small**) with integrated
**Grad-CAM++** explanations, **INT8 quantization** (dynamic + static PTQ), rigorous
**bootstrap confidence intervals**, and **correct efficiency benchmarking** — built
for deployment in low-resource clinical settings.

> **Research contribution** is not pneumonia classification alone, but the
> *integration*: explainable medical image classification + resource-efficient,
> quantized, deployment-oriented models + a reproducible medical-AI workflow.

> 🔬 **Scientific integrity.** This project **never fabricates results**. There are
> no simulated radiologist surveys, no invented metrics, and no `tracemalloc`-based
> "0.1 MB RAM" claims. Numbers appear in the thesis only after you run the pipeline;
> they are injected from `results/` by `scripts/render_report.py`.

---

## Table of Contents
- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start (synthetic smoke test)](#quick-start-synthetic-smoke-test)
- [Dataset](#dataset)
- [Full reproduction](#full-reproduction)
- [What gets produced](#what-gets-produced)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Limitations](#limitations)
- [Citation](#citation)

## Highlights
- **Three architectures, one pipeline** — fair, publishable baseline comparison.
- **Integrated Grad-CAM++** for correct / false-positive / false-negative cases.
- **Dynamic vs static INT8 PTQ** — a transparent compression study (static PTQ
  quantizes convolutions, the deployment-relevant win).
- **Honest benchmarking** — on-disk size, warmup-corrected latency (mean/std/p95),
  throughput, and sampled peak process RSS.
- **Statistical rigor** — bootstrap 95% CIs for accuracy and ROC-AUC; Youden-J
  threshold; per-error analysis.
- **Reproducible by construction** — YAML configs, global seeding, deterministic
  data loading, per-run `metadata.json`, and auto-generated CSV/Markdown/LaTeX
  tables and PNG/PDF figures.

## Repository structure
```
.
├── configs/            # base + per-model YAML (efficientnet_b0, resnet18, mobilenetv3)
├── data/               # raw/ processed/ splits/ (+ data/README.md; datasets not committed)
├── src/                # all logic (importable, typed, tested) — no notebook-only code
│   ├── config.py utils.py dataset.py transforms.py models.py metrics.py
│   ├── train.py evaluate.py inference.py quantize.py benchmarking.py
│   └── explainability.py visualization.py reporting.py
├── scripts/            # make_synthetic_data.py, run_smoke_test.sh, render_report.py
├── tests/              # pytest suite (unit + end-to-end integration)
├── results/            # figures/ tables/ metrics/ gradcam/ confusion_matrices/ roc_curves/
├── reports/            # thesis.md (results-driven), limitations.md, future_work.md, REPRODUCE.md
├── paper_assets/       # IEEE paper skeleton + captions + result summary
├── notebooks/          # thin demo notebook that calls into src/
├── Makefile  pyproject.toml  requirements.txt  environment.yml
└── LICENSE  CITATION.cff  CONTRIBUTING.md
```

## Installation
```bash
make setup            # creates .venv and installs pinned dependencies
# or manually:
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```
Reference environment: Python 3.12, PyTorch 2.12, torchvision 0.27 (CPU verified on
macOS arm64). For CUDA, install the matching `torch` build from pytorch.org.

## Quick start (synthetic smoke test)
Verify the entire pipeline end-to-end **without the dataset** — uses tiny synthetic
images (random; *not* research data):
```bash
make smoke
```
This runs validate → train (3 models) → evaluate → benchmark → quantize → Grad-CAM++
→ reporting and writes artifacts under `results/` and `models/`.

## Dataset
Download the **Kermany Chest X-Ray Pneumonia** dataset into `data/raw/chest_xray/`
(`train/` and `test/`, each with `NORMAL/` and `PNEUMONIA/`). Full instructions:
[`data/README.md`](data/README.md). Then:
```bash
make validate-data    # class balance, duplicate detection, train/val/test leakage check
```

## Full reproduction
```bash
make reproduce        # validate -> train (x3) -> evaluate -> benchmark -> quantize -> explain -> report -> render
```
Or stage by stage:
```bash
make train            # python -m src.train --config configs/efficientnet_b0.yaml
make train-baselines  # resnet18 + mobilenetv3
make evaluate         # test metrics + CIs + confusion/ROC/PR figures
make benchmark        # size/latency/throughput/RSS
make quantize         # dynamic + static INT8 + ONNX export
make explain          # Grad-CAM++ overlays
make report           # Tables 1-6 (CSV/MD/LaTeX) + limitations/future-work/paper assets
make render           # fill reports/thesis.md placeholders -> reports/thesis_rendered.md
```
Every stage is also a plain CLI, e.g.:
```bash
python -m src.train --config configs/resnet18.yaml --override train.epochs=15 data.batch_size=16
python -m src.inference --checkpoint models/efficientnet_b0_best.pth --image path/to/cxr.jpeg
```

## What gets produced
| Output | Location |
|---|---|
| Trained checkpoints, FP32/INT8 models, ONNX | `models/` |
| Metrics + CIs, history, metadata, error cases (JSON/CSV) | `results/metrics/` |
| Tables 1–6 + literature template (CSV + Markdown + LaTeX) | `results/tables/` |
| Training curves, ROC, PR, confusion matrix, quantization charts (PNG + PDF) | `results/figures/`, `results/roc_curves/`, `results/confusion_matrices/` |
| Grad-CAM++ overlays (correct / FP / FN) | `results/gradcam/` |
| Results-driven thesis + limitations + future work | `reports/` |
| IEEE paper skeleton + captions + summary | `paper_assets/` |

## Reproducibility
- `seed_everything()` seeds Python/NumPy/PyTorch and makes DataLoaders deterministic
  (seeded generator + worker init).
- Every training run writes a `metadata.json` manifest: timestamp, git SHA, seed,
  library versions, full config, and dataset statistics.
- All experiment behaviour is configured via `configs/` (no hidden constants).
- Thesis/paper numbers are injected from result files — never hand-typed.

## Testing
```bash
make test             # 40 unit + integration tests (CPU, offline, synthetic fixtures)
```
Covers config loading, seeding/determinism, dataset integrity (dedup + leakage),
transforms, all three models, metrics + bootstrap CIs, quantization (dynamic +
static PTQ), and a full train→evaluate→benchmark integration run.

## Limitations
Stated explicitly in [`reports/limitations.md`](reports/limitations.md): single
dataset, no external validation, no real-world/clinician validation (and none
simulated), no edge-hardware validation, and potential dataset bias. This is a
research artifact and **not a medical device**.

## Citation
If you use this work, please cite it (see [`CITATION.cff`](CITATION.cff)) and the
Kermany dataset [22 in `reports/thesis.md`].

## License
MIT (code only) — see [`LICENSE`](LICENSE). The Kermany dataset has its own CC BY 4.0
terms.
