# External-dataset work plan and current status

## What is already complete

The repository contains a real RSNA Pneumonia Detection Challenge probe. It uses
inference only, maps `Normal` to NORMAL and `Lung Opacity` to PNEUMONIA, excludes
`No Lung Opacity / Not Normal`, and keeps the original locked threshold. It is
supplementary domain-shift evidence, not clinical validation.

Run it with:

```bash
make docker-external-rsna
```

The main output is `results/metrics/rsna_external_metrics.json`.

## NIH ChestX-ray14 and CheXpert

Both are possible additions, but they are not interchangeable with the current RSNA
probe:

- NIH ChestX-ray14 is an adult, multi-label dataset. “Pneumonia” is one label among
  many findings, so the paper must define how uncertain and missing labels are handled.
- CheXpert uses uncertainty labels (`-1`) and requires a pre-specified policy for
  mapping or excluding them. It is also much larger than the current primary data.

The current repository does not contain either dataset, so no NIH or CheXpert result is
claimed. Adding them requires downloading the data under their own terms of use,
recording source/version/checksums, converting only the pneumonia/normal subset, and
running the existing inference-only evaluator without changing the threshold or
checkpoint.

For the minimum Nepal-journal paper, one well-documented external dataset (RSNA) is
more defensible than adding two datasets with rushed label mappings. NIH and CheXpert
should be added only if their data terms, preprocessing, label policy, and resulting
metrics can be documented completely.
