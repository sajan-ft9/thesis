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
- All three models' single-split results are corroborated by 5-fold cross-validation
  (5 independent retrains each, every fold evaluated on the same held-out test set):
  mean AUC 0.9707±0.0057 (EfficientNet-B0), 0.9608±0.0084 (ResNet-18), 0.9700±0.0159
  (MobileNetV3-Small) — all within ~0.01 of their single-split AUC. Notably,
  MobileNetV3-Small's fold-to-fold AUC standard deviation is 2.8x EfficientNet-B0's,
  adding evidence (beyond §4.2.1's significance tests) that its single-split AUC
  advantage is not a stable property of the model. Patient-level grouping across folds
  was not verified (fold assignment is image-level; see item 1) and only a single CV
  seed (5-fold, seed 42) was used — a multi-seed x multi-fold study remains future work.
- Static INT8 quantization is backend-dependent; if unsupported on the host, the
  pipeline reports this rather than substituting estimated numbers.
