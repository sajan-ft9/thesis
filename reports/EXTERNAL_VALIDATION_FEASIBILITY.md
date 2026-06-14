# Feasibility Analysis — NIH ChestX-ray14 External Validation

**Role:** Senior Medical AI Researcher / IEEE Reviewer / Research Software Engineer
**Date:** 2026-06-14 · **Status:** analysis & recommendation (NOT an implementation)
**Priority directive:** thesis completion over experimentation; a small, defensible
external validation is preferred over a large, incomplete one.

---

## 1. Executive Summary

External validation on NIH ChestX-ray14 is **technically very feasible** — the existing
preprocessing, model output, and metric/figure code are directly reusable, and only a
small *test-only* evaluation path needs to be added (the current `evaluate` path requires
a train directory). The dominant cost and risk are **not code** but **(a) data
acquisition** (ChestX-ray14 is ~42 GB; pneumonia images are rare and scattered) and
**(b) scientific framing** (ChestX-ray14 pneumonia labels are NLP-mined and noisy, and
the population is adult vs. Kermany's pediatric — so a performance drop is expected and
could be *misread* as model failure if framed poorly).

**Verdict: RECOMMENDATION B — proceed only if it fits inside a ~1-day time-box, gated on
smooth credential-free data access; otherwise defer to future work (C).** The scientific
value is high (it removes the single-dataset limitation, the #1 reviewer concern), but it
is only worth doing if it stays cheap and is framed as an *exploratory zero-shot
generalization probe under domain shift and label noise* — not as a clean generalization
claim.

---

## 2. Feasibility Assessment (grounded in the code)

| # | Current assumption (verified in code) | NIH compatibility |
|---|---|---|
| 1 | Dataset = folders `<root>/<split>/{NORMAL,PNEUMONIA}/*` (`scan_split`) | ✅ Reproducible by arranging an NIH subset into the same folder layout (symlinks/copies). |
| 2 | Label map `{"normal":0, "pneumonia":1}`; extensions include `.png` | ✅ NIH is PNG; mapping works once images are foldered as NORMAL / PNEUMONIA. |
| 3 | Preprocessing = `Image.open().convert("RGB")` → Resize(224) → Normalize(ImageNet) | ✅ Fully compatible: grayscale 1024² NIH PNGs convert to 3-channel and resize cleanly. **No preprocessing change.** |
| 4 | Model output = single sigmoid logit (pneumonia probability) | ✅ Binary pneumonia-vs-normal inference is exactly the model's task. |
| 5 | Checkpoint `models/efficientnet_b0_best.pth` (inference only) | ✅ Loaded by `load_checkpoint`; `collect_predictions` runs inference with no retraining. |

**Binary evaluation possible?** Yes. NIH provides "No Finding" (→ Normal) and a
"Pneumonia" label (→ Pneumonia). A clean binary Normal-vs-Pneumonia subset can be built
by selecting those two groups and excluding all other-pathology images.

**One genuine code gap:** `evaluate_checkpoint()` → `build_dataloaders()` expects **both**
`train/` and `test/` directories (it performs the stratified split + dedup). An
external, test-only set therefore needs a **small new function** (≈30–40 lines) —
`evaluate_directory(checkpoint, image_dir)` — that reuses the *existing* helpers
`ChestXRayDataset.from_directory`, `collect_predictions`, `compute_metrics`,
`metric_confidence_intervals`, `plot_confusion_matrix`, `plot_roc_curve`,
`plot_pr_curve`. No change to any existing function or result.

---

## 3. Estimated Implementation Effort

**Overall: MEDIUM (1–3 days)** — but decomposes into a LOW code part and a variable data part.

| Component | Effort | Why |
|---|---|---|
| New `evaluate_directory` path + NIH manifest builder (parse `Data_Entry_2017.csv`, select/arrange Normal & Pneumonia, balanced sample) | **LOW (~0.5 day)** | All metric/figure/inference helpers already exist; only glue + a CSV filter. |
| **Data acquisition** | **LOW→HIGH (swing factor)** | Full ChestX-ray14 ≈ 42 GB (12 archives); pneumonia (~1.4k images) is rare and scattered across all archives, so a true "all pneumonia" set needs the full download. A **credential-free HuggingFace mirror** (e.g. `alkzar90/NIH-Chest-X-ray-dataset`) with label filtering would make this LOW–MEDIUM (as the Kermany HF mirror did); manual 42 GB tar handling is HIGH (hours–day of bandwidth). Disk is not a blocker (226 GB free). |
| Verification + thesis subsection + careful framing | **LOW (~0.5 day)** | One new isolated results subsection + caveats. |

**Becomes LOW (<1 day)** *iff* a credential-free mirror yields a balanced
pneumonia/no-finding subset quickly. **Becomes HIGH** if it requires downloading and
untarring the full 42 GB corpus. This uncertainty is exactly why the recommendation is
time-boxed (Section 7).

---

## 4. Label Quality Review

*(Counts below are the commonly-cited ChestX-ray14 figures from Wang et al., 2017;
confirm exactly against `Data_Entry_2017.csv` before use.)*

1. **Pneumonia cases:** ~**1,431** images carry the "Pneumonia" label (one of the rarest
   of the 14 findings).
2. **"No Finding" cases:** ~**60,361** images (abundant → can be sub-sampled to balance).
3. **Class-definition compatibility with Kermany:** *Partial.* Both have a normal class
   and a pneumonia class, but: (a) NIH is **adult**, Kermany is **pediatric** (1–5 yr) —
   a real domain shift; (b) NIH labels are **NLP-mined from reports** (noisy; the
   pneumonia label in particular is known to be unreliable — Oakden-Rayner, 2020), while
   Kermany labels are expert-verified; (c) NIH is **multi-label**.
4. **Filtering required:** *Yes.* Keep only "No Finding" (→ Normal) and pneumonia-positive
   images (→ Pneumonia); **exclude** all images that are neither, to preserve the binary
   Normal-vs-Pneumonia task the model was trained for.
5. **Multi-pathology samples:** Pneumonia frequently co-occurs with Infiltration,
   Effusion, etc. **Simplest defensible rule:** *pneumonia-inclusive* — an image counts
   as Pneumonia if "Pneumonia" appears anywhere in its finding list (matches the clinical
   "is pneumonia present?" screening question). Do **not** require pneumonia to be the
   sole finding (that would cherry-pick easy cases and shrink n).

**Simplest defensible strategy:** Normal = "No Finding" (exact); Pneumonia =
pneumonia-inclusive; balance by taking all ~1,431 pneumonia + an equal seeded random
sample of "No Finding" (~1,431) → a clean, ~2,800-image binary external test set
(optionally one image per patient to reduce correlation).

> **RSE/reviewer note (higher-value alternative):** if the goal is simply to *demonstrate
> external evaluation cleanly*, the **RSNA Pneumonia Detection** set (also NIH-derived but
> with **expert-adjudicated** pneumonia labels) has substantially lower label noise than
> ChestX-ray14's NLP labels, for similar effort. ChestX-ray14 remains valid but carries a
> heavier "is the drop the model or the labels?" caveat.

---

## 5. Minimal Validation Strategy (if GO)

Smallest scientifically valid experiment — **inference only**:

- Use `models/efficientnet_b0_best.pth` as-is. **No retraining, no fine-tuning, no
  threshold tuning** (use the same fixed 0.5 threshold; optionally also report the
  Kermany-derived Youden threshold, clearly labelled, without re-optimising on NIH).
- Build the balanced ~2,800-image Normal/Pneumonia subset (Section 4).
- Run `evaluate_directory` → report **ROC-AUC, Accuracy, Precision, Recall/Sensitivity,
  Specificity, F1** with bootstrap 95% CIs, plus a **ROC curve** and **confusion matrix**.
- Add **one isolated thesis subsection** "§4.x External Validation (Exploratory,
  Zero-Shot)" — additive only; **do not touch any existing Kermany result, table, or
  figure.**

Expected outcome: a **measurable performance drop** (literature on pediatric→adult /
ChestX-ray14 transfer typically sees AUC fall into the ~0.6–0.8 range). Per the directive,
**a drop is an acceptable and informative result** — it demonstrates that external
evaluation was performed and quantifies the domain-shift gap.

---

## 6. Risk Analysis

1. **Timeline risk — MEDIUM.** Data acquisition can balloon (42 GB / slow mirror) and eat
   days. *Mitigation:* hard time-box + credential-free mirror + fall back to future work.
2. **Scientific-validity risk — MEDIUM/HIGH.** ChestX-ray14 pneumonia labels are noisy and
   the population differs, so a low score conflates **domain shift + label noise + model**.
   *Mitigation:* frame strictly as *exploratory zero-shot generalization*; state the label-
   noise caveat; consider RSNA as a cleaner cross-check.
3. **Publication-readiness risk — LOW (net positive).** Adding *any* honest external
   validation directly addresses the most common reviewer rejection reason; even a drop,
   framed correctly, strengthens the paper and PhD applications.
4. **Thesis-conclusion risk — LOW *if isolated*.** The thesis claims efficient, explainable,
   reproducible performance **on Kermany** — not universal generalization. An external
   drop does not contradict it, *provided* the result is additive and not over-claimed.

**Could NIH validation WEAKEN the thesis? — YES, if done incorrectly**, specifically by:
(a) sloppy label mapping (e.g. treating every non-pneumonia image as "normal") → garbage
metrics; (b) too few pneumonia images → an unstable AUC; (c) framing a domain-shift drop
as a defect of the proposed method; or (d) letting the external run modify/contaminate the
existing Kermany numbers. **Done correctly** (clean balanced subset, zero-shot, honest
"exploratory under domain shift + noisy labels" framing, fully isolated section), it
**strengthens** the thesis and is a clear asset for PhD applications.

---

## 7. Recommended Approach & Go/No-Go

### RECOMMENDATION B — Proceed only if it fits a ~1-day time-box (else defer).

**Rationale.** Code is LOW and the scientific/publication value is HIGH, but data
acquisition is uncertain and the framing risk is real; the user's stated priority is
finishing the thesis. So gate it:

**GO (do it now) if** a credential-free NIH mirror yields a balanced pneumonia/no-finding
subset within a few hours **and** the `evaluate_directory` path is the only code added.
Then run inference-only, add one isolated subsection, run `make verify-numbers`.

**NO-GO (defer to future work) if** acquiring enough pneumonia images requires the full
42 GB manual download, the mirror stalls, or it threatens the submission timeline. In that
case, keep the existing, well-written "external validation" entry in §5.3 Limitations /
§6.2 Future Work — which is already an acceptable position for a Master's thesis.

**If GO, consider RSNA Pneumonia instead of / in addition to ChestX-ray14** for lower
label noise at comparable effort.

### Go/No-Go decision: **CONDITIONAL GO (time-boxed, ≤1 day, data-access-gated).**

| Dimension | Assessment |
|---|---|
| Feasibility (code) | High — reuses existing pipeline; one ~40-line function |
| Feasibility (data) | Uncertain — the deciding factor; gate on a credential-free mirror |
| Implementation effort | MEDIUM overall (code LOW, data the swing) |
| Scientific value | High *if framed as exploratory zero-shot*; otherwise risky |
| Risk to thesis | Low if isolated & honest; real if sloppy or over-claimed |
| **Recommendation** | **B — conditional, time-boxed GO; else defer (C)** |

**Bottom line:** Worth doing *only* as a small, isolated, honestly-framed, inference-only
probe that stays within a day. If data access isn't quick and clean, the thesis is already
defensible without it — leave it as the clearly-stated future work it currently is.
