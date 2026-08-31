# Supervisor Progress Brief

**Student:** Sajan · **Date:** 2026-06-14 · **Project:** Memory-Efficient, Explainable,
Quantized Pneumonia Detection from Chest X-rays for Resource-Constrained Healthcare

---

## 1. One-line summary
A reproducible pipeline that trains a lightweight **EfficientNet-B0** (vs ResNet-18 /
MobileNetV3-Small baselines), explains it with **Grad-CAM++**, compresses it with
**INT8 quantization**, and reports **honest accuracy, statistics, and memory** on the
Kermany pediatric CXR dataset.

## 2. Objectives and status
| Objective | Target | Result | Status |
|---|---|---|---|
| Diagnostic performance | AUC ≥ 0.95 | 0.968 (95% CI 0.950–0.982) | ✅ |
| Sensitivity (screening) | ≥ 0.93 | 0.967 | ✅ |
| Model size | < 50 MB | 5.22 MB (INT8 static) | ✅ |
| CPU latency | < 100 ms | 16 ms (INT8) / 133 ms (FP32) | ✅ via quantization |
| Inference-time RAM increase | < 2 GB | 39 MB (INT8) / 1156 MB (FP32), medians of 5 runs | ✅ |
| Explainability | Grad-CAM++ (correct/FP/FN) | 12 overlays | ✅ qualitative |
| Reproducibility | seeded + tested | 40 tests, manifests, `make reproduce` | ✅ |

## 3. Key results (held-out test, n=624; all reproducible)
- **EfficientNet-B0:** AUC 0.968, sensitivity 0.967, specificity 0.838, F1 0.937;
  best precision/specificity balance of the three models (38 false positives vs 72/77).
- **Quantization:** static INT8 → **70.5% smaller, ~8× faster CPU, ~30× lower inference-time
  memory increase** (median of 5 isolated runs), for a 2.3-point AUC cost; dynamic INT8 keeps
  accuracy but barely compresses and gives no memory benefit.
- **Memory:** streaming loader uses **~10.5× less RAM** than naïve full-dataset loading
  (medians of 5 runs each; see reports/thesis_final.md §4.4.1 for the measurement-reliability
  fix behind these numbers).
- **Data integrity:** automatically detected and removed **26 duplicate images** causing
  train/validation leakage; test set kept canonical.

## 4. What to show in the meeting
1. **This brief** (1 page) — lead with it.
2. **`reports/thesis_final.md`** — the written thesis with real numbers.
3. **The repository** — `README.md`, then `make test` (40 pass) and `make reproduce` to
   show it is fully reproducible; figures in `results/`.
4. **`reports/related_work_annotated.md`** — the 10 papers you have read/reviewed.
5. **`paper_assets/paper_ieee.tex`** — the conference-paper draft built from the results.
6. **`reports/RESEARCH_PLAN.md`** — proposed next steps (shows you can scope ahead).

## 5. Honest framing (what is and isn't novel)
- The *engineering and rigour* are strong: reproducible, tested, leakage-controlled,
  correctly benchmarked (memory included), no fabricated results.
- The *diagnostic accuracy* is in line with recent lightweight/edge work
  (Benmalek 2025, PneuNet 2026, Nettur 2025) — not a record, and we do not claim one.
- The contribution is the **honest, integrated, deployment-and-memory-focused evaluation**,
  not a new architecture.

## 6. Proposed next steps (pick with supervisor)
- **Lowest effort, high value:** external validation on NIH ChestX-ray14 / RSNA to address
  the single-dataset limitation.
- **Medium:** quantization-aware training (QAT) to close the static-PTQ accuracy gap;
  quantitative Grad-CAM++ localization vs RSNA bounding boxes.
- **Hardware:** on-device (Raspberry Pi / Jetson) latency + energy benchmarking.
- Full plan in `reports/RESEARCH_PLAN.md`.

## 7. Questions for the supervisor
1. Target venue — Master's thesis only, or also a conference (e.g., IEEE Access / EMBC /
   ISBI)? This decides how much of §6 to pursue.
2. Is single-dataset evaluation acceptable for the thesis, or is external validation
   required before submission?
3. Is the current scope (no real edge hardware, CPU proxy) acceptable, or should I
   prioritise a Jetson/Pi deployment?
4. Citation style and any required thesis template/format for the institution.

## 8. Likely defense questions — and answers to have ready
- **"Why EfficientNet-B0 when MobileNetV3-Small has higher AUC?"** MobileNetV3-Small wins
  on AUC by 0.6 points but doubles false positives (77 vs 38; specificity 0.67 vs 0.84).
  For a *screening* tool clinicians must trust, the false-alarm rate dominates, so
  EfficientNet-B0 (best F1/specificity balance) is the default; MobileNetV3-Small is the
  minimum-footprint alternative. (See thesis §4.2.)
- **"What's novel — pneumonia detection is solved?"** The contribution is the
  *methodology*, not the classifier: reproducible, integrity-checked (leakage removed),
  deployment- and memory-honest evaluation with a real quantization study — things most
  CXR papers (incl. the closest, Benmalek 2025) omit. Lead with this, not accuracy.
- **"Only one dataset?"** Acknowledged as the main limitation (§5.3); acceptable for a
  Master's thesis; external validation (NIH/CheXpert) is the first future-work item.
- **"Did you deploy on edge hardware?"** No — CPU/MPS is used as an explicit *proxy*;
  the thesis says "suitable for" deployment, and on-device benchmarking is future work.
- **"Are the numbers real?"** Yes — every figure/number is regenerated by the code from
  raw data (`make reproduce`); `python scripts/verify_thesis_numbers.py` checks the
  headline numbers against `results/metrics/*.json`. Nothing is simulated.
