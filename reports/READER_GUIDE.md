# Reader Guide

## What the thesis claims

The thesis does not claim a new state-of-the-art classifier or clinical readiness. Its central claim is narrower: a lightweight CNN can be evaluated more honestly when data integrity, uncertainty, qualitative visual error analysis, quantization, and measured resource cost are reported together.

The primary result is EfficientNet-B0 on a held-out portion of the Kermany pediatric chest-X-ray dataset: AUC 0.9678, sensitivity 0.9667, specificity 0.8376, and F1 0.9366. The dataset is imbalanced toward pneumonia, so accuracy alone is not sufficient.

The external RSNA result is supplementary and zero-shot: AUC 0.8892, sensitivity 0.9553, specificity 0.6060. It shows partial transfer under pediatric-to-adult and acquisition/domain shift, but it is not clinical validation.

Static INT8 reduces serialized model size from 17.67 MB to 5.22 MB and isolated inference RSS from 544.4 MB to 222.7 MB, while measured quantization-protocol CPU latency increases from 104.5 ms to 152.7 ms, AUC falls from 0.9678 to 0.9427, and specificity falls to 0.4701 at the fixed threshold. That is a trade-off, not an unconditional improvement.

## How to read the evidence

- Treat Kermany results as image-level, in-dataset benchmark evidence, not patient-level or hospital-level performance.
- Treat RSNA as an exploratory generalization probe; its simplified label mapping and balanced subset limit what can be concluded.
- Treat Grad-CAM++ images as visual hypotheses for error analysis, not validated explanations.
- Treat memory and speed numbers as measurements on one workstation configuration, not guaranteed edge-device performance.
- Statistical tests compare model outputs on the same test cases. The EfficientNet-vs-comparator AUC differences are not significant in the committed DeLong results; the ResNet-vs-MobileNet comparison is significant at uncorrected p=0.01, so “all three are indistinguishable” is too broad.

## What is solid and what remains open

Solid within the repository: the headline thesis values match the generated JSON artifacts; the data-integrity report exposes the naïve train/validation duplicate problem; the test set is kept separate; and the pipeline has code and synthetic integration coverage.

Still unresolved: independent rerun from the licensed raw datasets, patient-level identity checks, near-duplicate detection, subgroup analysis, calibration on a genuinely held-out development population, repeated seeds, multi-centre external validation, radiologist reader studies, prospective evaluation, and testing on actual target hardware.

## Bottom line

The thesis is credible as a careful MSc research/software-engineering study if its conclusions stay scoped to the datasets and experiments performed. It would be misleading to describe it as clinically accurate, universally generalizable, independently validated across hospitals, or proven “non-fabricated” solely because its files are internally consistent.
