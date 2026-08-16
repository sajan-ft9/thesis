# Thesis Audit

## Verdict

**Overall: substantially supported as a research artifact, but not clinically validated and not fully reproducible from the repository alone.** I found no obvious fabricated survey, invented trust score, or impossible headline number. The committed thesis numbers match the committed metric JSON files (`python3 scripts/verify_thesis_numbers.py`: 31 checks passed). This is internal consistency, not independent proof of every experiment.

## Evidence checked

- Source pipeline exists for dataset checks, training, evaluation, quantization, benchmarking, memory profiling, Grad-CAM++, reporting, and RSNA inference-only evaluation.
- `results/metrics/` contains checkpoints, metadata, metrics, and generated figures; manifests record seed, Git SHA, environment, and configuration.
- The data report records 26 duplicate training files and 9 contents crossing the naïve train/validation split; the post-dedup working split is clean by byte-content hash.
- The RSNA artifact contains 12,024 balanced images and explicitly records no retraining or threshold tuning.
- Official RSNA documentation describes expert annotation and the three categories used by the thesis mapping; official FDA material treats software analyzing medical images as a distinct regulatory/clinical category. See [RSNA dataset page](https://www.rsna.org/education/ai-resources-and-training/%20%5C%20ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018), [RSNA annotation paper](https://pubs.rsna.org/doi/10.1148/ryai.2019180041), and [FDA clinical decision support guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software).

## Important corrections made

1. `reports/limitations.md` incorrectly said there was no external validation. It now describes the single exploratory RSNA probe.
2. “26 duplicates caused train/validation leakage” was too compressed. The corrected wording distinguishes 26 duplicate files from the 9 duplicate contents that crossed the naïve split.
3. “The three architectures are statistically indistinguishable on AUC” was too broad. The pairwise results show EfficientNet is not significantly different from either comparator, while ResNet vs MobileNet has uncorrected DeLong p=0.01.
4. README language now says the artifacts are internally consistent rather than asserting that fabrication has been disproven.

## Missing or weak steps

### High priority

- Reproduce from a documented raw-data checksum and a clean environment; the repository does not contain the raw Kermany/RSNA data.
- Add patient-level identifiers or a defensible patient-overlap check. File-level MD5 checks cannot establish patient independence.
- Run repeated seeds and report variability; one seed is weak evidence for model ranking.
- Pre-specify the primary model/metric and correct for multiple pairwise statistical tests.
- Add a true development/calibration protocol if threshold or calibration claims are retained. A threshold computed on RSNA must remain clearly descriptive, never deployment-tuned evidence.

### Required before clinical or deployment claims

- Multi-centre, age-diverse external validation with a fixed protocol.
- Prospective or retrospective reader study against qualified radiologists and a documented reference standard.
- Subgroup/fairness analysis, missing-data and image-quality handling, failure review, and uncertainty reporting.
- Actual target-device benchmarks including cold start, preprocessing, throughput, power, thermal behavior, and memory limits.
- Regulatory, privacy, cybersecurity, and clinical workflow review. The current work explicitly does not provide these.

## Final assessment

The defensible conclusion is: **the project demonstrates a reproducible, leakage-aware, resource-measured prototype and an exploratory cross-dataset transfer result.** The indefensible conclusions would be: “clinically proven,” “safe to diagnose patients,” “generalizes to hospitals,” “explainable because Grad-CAM looks plausible,” or “non-fabrication proven by a number-check script.”
