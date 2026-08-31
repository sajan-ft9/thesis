# Limitations

These limitations are stated explicitly and must not be hidden in the thesis or paper.

1. **Single-dataset evaluation.** All training and testing use the Kermany
   pediatric chest X-ray dataset from a single institution. Results may not
   transfer to other scanners, acquisition protocols, or hospitals.

2. **No external validation.** The model has not been evaluated on an independent
   external dataset (e.g. NIH ChestX-ray14, CheXpert). Reported metrics reflect
   in-distribution performance only.

3. **No real-world clinical validation.** No prospective study or reader study
   with practising radiologists was conducted. Grad-CAM++ outputs are model-derived
   visualisations, not clinically validated explanations. (No radiologist ratings
   are simulated anywhere in this project.)

4. **No edge-hardware validation.** Efficiency metrics are measured on a
   general-purpose CPU as a proxy for edge devices. Latency, memory and power on
   actual Raspberry Pi / Jetson hardware were not measured.

5. **Potential dataset bias.** The Kermany dataset is pediatric and class-imbalanced
   (pneumonia-heavy). Performance across age, sex, ethnicity and comorbidity
   subgroups was not assessed and may be biased.

Additional methodological notes:
- The official Kermany validation split (16 images) is too small for model
  selection; a stratified split carved from the training set is used instead, with
  the official test set held out untouched.
- The primary model's single-split result is corroborated by 5-fold cross-validation
  (5 independent retrains, each evaluated on the same held-out test set): mean AUC
  0.9707 (std 0.0057), within 0.003 of the single-split AUC (0.9678). This was not
  done for the two baseline architectures, and patient-level grouping across folds
  was not verified (see item 1).
- Static INT8 quantization is backend-dependent; if unsupported on the host, the
  pipeline reports this rather than substituting estimated numbers.
