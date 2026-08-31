# Small Model, Big Lungs

### Can a tiny AI spot pneumonia on an X-ray — without a supercomputer?

A plain-English walkthrough of the thesis *"Memory-Efficient, Explainable, Quantized Pneumonia Detection,"* checked line-by-line against the actual code and data before being summarized here.

✅ Numbers independently re-checked
✅ No fabricated results found
⚠️ Single dataset — read the caveats

---

## 1. What is this thesis actually about?

It trains a small AI model to look at a chest X-ray and flag pneumonia — then spends most of its effort proving that the model is **small enough**, **honest enough**, and **explainable enough** to trust that claim.

Three words matter here, and each is a design goal, not marketing:

- **Efficient** — the model has to run on ordinary hardware (a normal laptop CPU, no graphics card), and use as little memory as possible.
- **Explainable** — it doesn't just say "pneumonia: yes/no." It produces a heatmap showing *which part of the lung* it looked at, so a clinician can sanity-check the AI's reasoning instead of trusting a black box.
- **Honest** — the thesis treats "did we measure this correctly?" as a research question in its own right: duplicate images, data leakage, and mis-measured memory usage are actively hunted down and reported, not swept under the rug.

## 2. Why does this matter?

Pneumonia is one of the leading causes of death in young children worldwide, and the burden falls hardest on places with the fewest resources — clinics with no radiologist on staff, and often no GPU either. A model that only works on expensive server hardware is useless there.

Most AI research chases the highest possible accuracy with the biggest possible model. This thesis asks the opposite question: **how small can the model get, and how much can be verified about it, before it stops being useful?** That reframing — from "best score" to "best score you can actually trust and actually deploy" — is the thesis's real contribution.

## 3. How does it work, step by step?

1. **Clean the data first.** Before training anything, the pipeline scans the X-ray dataset for duplicate images and images that leaked between the training and validation sets — the kind of silent bug that makes a model look better than it is. It found and removed **26 duplicates** and **9 leaked images**.
2. **Race three candidate models.** Three well-known lightweight architectures — EfficientNet-B0, ResNet-18, and MobileNetV3-Small — are trained on identical data with an identical procedure, so the comparison is fair.
3. **Pick a winner by pre-agreed rules.** Rather than just picking whichever model scores highest on one number (which can be cherry-picking in disguise), the winner is chosen against criteria decided *before* looking at results: overall accuracy, how well-calibrated its confidence is, and how many patients it wrongly flags as healthy or sick.
4. **Shrink it (quantization).** The winning model is compressed using INT8 quantization — storing its internal numbers with less precision, the way a photo can be saved at lower resolution. Done carelessly, this technique can break the model entirely; the thesis documents exactly how much precision can be sacrificed before that happens.
5. **Make it explain itself.** Grad-CAM++ generates a heatmap overlay on each X-ray showing which pixels most influenced the model's decision — including on cases where the model was *wrong*, which is often more informative than the cases where it was right.
6. **Test it on a different hospital's data.** To see whether the model generalizes beyond its training population, it's also run — without any retraining — on a second, independent dataset (RSNA, adult patients) as a stress test.
7. **Retrain everything 5 times to be sure.** All three models — not just the chosen one — are retrained from scratch on 5 different slices of the training data and tested each time on the same untouched test set, so the headline result can't be dismissed as one lucky split, and neither can the runner-up's.
8. **Grade its own homework.** A verification script re-derives every headline number in the write-up straight from the raw result files and fails loudly if even one doesn't match — which is the mechanism behind the "no fabricated numbers" claim below.

## 4. What did they actually find?

On **624 held-out test X-rays**, the chosen model (EfficientNet-B0):

| Metric | Value |
|---|---|
| AUC — overall discrimination ability | 96.8% |
| Sensitivity — true pneumonia cases correctly caught | 96.7% |
| Specificity — healthy X-rays correctly cleared | 83.8% |
| Overall accuracy | 91.8% |

After shrinking the model with INT8 quantization:

| Metric | Result |
|---|---|
| Size | 70.5% smaller on disk (17.7 MB → 5.2 MB) |
| Speed | ~8× faster per prediction (133 ms → 16 ms) |
| Memory | ~30× less inference-time memory (1156 MB → 39 MB, median of 5 runs) |
| Accuracy cost | −2.3 points AUC |

**A real, non-obvious finding:** quantizing the whole model at one uniform precision ("per-tensor") **broke it** — accuracy collapsed to a coin-flip. Quantizing each channel separately ("per-channel") fixed it completely. That's a genuine engineering insight, not a footnote.

On the second hospital's dataset (no retraining), AUC dropped to **88.9%** — a real and expected sign of domain shift, reported honestly rather than hidden.

## 5. Is it legit, or is this "AI thesis" hand-waving?

It checks out. Before writing this guide, every headline number was traced back to the raw result file it came from — not just taken on faith from the write-up.

**What was actually verified:** the thesis's own verification script (`verify_thesis_numbers.py`) recomputes every reported number from the raw JSON result files and compares it to what's written in the final report — **31 out of 31 numbers matched exactly**. Several of those numbers (test AUC, external-validation AUC, duplicate-image count, dataset split sizes) were then spot-checked a second, independent way by opening the raw JSON files directly. The automated test suite runs clean.

Just as tellingly: the project's own history shows a **correction**, not a cover-up — an earlier, inflated cross-validation number and an earlier, wrongly-measured memory figure were caught and fixed, with the fix documented rather than quietly erased. That's what good scientific hygiene looks like in practice, and it's stronger evidence of trustworthiness than a suspiciously clean story would be.

Even the memory numbers above were caught mid-correction: a first attempt at measuring peak memory used a background thread that sampled memory every 10 milliseconds, which could simply miss a short spike between samples — and did, swinging by over 400 MB between two runs of the *identical* configuration. The fix reads the operating system's own memory high-water-mark counter instead (which can't miss a spike) and reports the median of 5 independent runs rather than a single sample — which is why the memory figures above changed from the very first draft of this guide.

## 6. What this thesis does *not* claim

- **One primary dataset.** The main model is trained on a single pediatric X-ray collection; the second dataset is used only as a quick, exploratory stress test, not a full clinical validation.
- **Cross-validation splits are image-level, not patient-level.** All three models were retrained 5 separate times on different data splits (5-fold cross-validation) to check the headline results weren't a lucky split — they weren't. But the 5 splits were done image-by-image rather than confirmed patient-by-patient, so if the same patient appears in more than one image, they could in principle land in both the training and held-out portion of a fold. Also, only one way of splitting into 5 folds was tried (a different random split could give slightly different numbers) — both are stated as open limitations, not claimed as solved.
- **CPU numbers are a proxy.** Efficiency is measured on a standard laptop CPU, not on an actual low-power edge device like a Raspberry Pi.
- **Heatmaps are shown, not graded.** No radiologist reviewed whether the Grad-CAM++ heatmaps point at clinically meaningful regions — that's a qualitative illustration, not a validated diagnostic aid.
- **Not a medical device.** The thesis is explicit about this itself: this is a research prototype, not something cleared for clinical use.

## 7. Elevator pitch

**30-second version:** "I built a small AI model that spots pneumonia on chest X-rays accurately enough to be useful, shrank it to run fast on ordinary hardware with almost no accuracy lost, made it show its reasoning with heatmaps instead of being a black box, and then — this is the part I actually care about — built a system that automatically double-checks every number I publish against the raw results, so nothing in the write-up is guessed or exaggerated."

### If someone pushes back

**"Isn't a small model just worse?"**
A little — the fully-shrunk version loses about 2 accuracy points — but it becomes 8× faster and uses about 30× less memory during inference, which is the whole trade being studied, not an accident.

**"How do you know the numbers are real?"**
An automated script recomputes every reported number from the raw output files and fails if any of them don't match — it's not a manual promise, it's enforced by code.

**"Couldn't the good result just be a lucky train/test split?"**
Checked directly, for all three models: each was retrained from scratch 5 separate times on 5 different slices of the training data (5-fold cross-validation), and every version was tested on the same untouched test set. The chosen model's average result (96.7% AUC) landed almost exactly on its original single-run result (96.8%) — not a fluke. As a bonus, this check also revealed that the model with the slightly *higher* single-run score (MobileNetV3-Small, 97.4%) was actually the least consistent of the three across the 5 retrains — more evidence the chosen model was the right pick, not just the AUC leaderboard winner.

**"Would this work in a real hospital?"**
Not yet, and the thesis says so directly — it's evaluated on one main dataset with one exploratory cross-hospital check, not a clinical trial.

**"What's the actual contribution, if the model architecture isn't new?"**
The contribution is the evaluation discipline: fair head-to-head comparison, pre-registered model-selection criteria, leak detection, correct memory measurement, and a self-verifying results pipeline — rigor that's often missing from papers that only chase a headline accuracy number.

---

**Where to look in the repo** — full write-up: `reports/thesis_final.md` · defense reference: `reports/THESIS_HANDBOOK.md` · rebuild everything from scratch: `make reproduce` · check every number: `python scripts/verify_thesis_numbers.py` · re-run the 5-fold cross-validation: `python scripts/run_kfold_cv.py --config configs/<model>.yaml --folds 5`.
