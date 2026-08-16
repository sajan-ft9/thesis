# Publication Execution Checklist

## Phase 1 — Scope lock

- [ ] Adopt the narrow title and primary research question in `PUBLICATION_RECOMMENDATIONS.md`.
- [ ] Make EfficientNet-B0 + FP32/INT8 trade-off the primary result.
- [ ] Move RSNA to secondary/appendix status.
- [ ] Replace clinical/explainability overclaims with prototype, qualitative visualization, and exploratory language.

## Phase 2 — Code corrections

- [ ] Select threshold on validation only.
- [ ] Measure actual saved artifact size.
- [ ] Move RSS benchmark into fresh subprocesses.
- [ ] Record quantization qconfig and fallback status.
- [ ] Update `src/reporting.py` so generated limitations match the current thesis.
- [ ] Add tests for threshold provenance, artifact-size measurement, and qconfig reporting.

## Phase 3 — Experiments

- [ ] Run EfficientNet-B0 with seeds 42, 43, and 44.
- [ ] Run locked FP32 and static-INT8 evaluation for each seed.
- [ ] Run calibration-batch sensitivity analysis without touching the test set.
- [ ] Optionally run Grad-CAM++ occlusion/deletion faithfulness analysis.
- [ ] Preserve raw logs, manifests, checkpoints, JSON, and hardware details.

## Phase 4 — Manuscript

- [ ] Report image-level split and explicitly state that patient-level independence is unavailable.
- [ ] Report prevalence/class balance for every evaluation dataset.
- [ ] Report threshold selection, calibration split, bootstrap method, and multiple-testing policy.
- [ ] Report exact artifact sizes, backend, CPU, thread count, input size, warmup, repeats, and RSS protocol.
- [ ] Include CLAIM 2024 and TRIPOD+AI completed checklists as supplements where the journal permits.
- [ ] Add data/license/ethics/funding/conflict statements.

## Phase 5 — Final verification

```bash
make setup
make test
make lint
make verify-numbers
git diff --check
```

Before submission, verify that every number in the manuscript is generated from a JSON artifact and that the manuscript does not call this a clinical device, reader study, patient-level validation, or physical edge deployment.
