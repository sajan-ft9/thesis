# Limitations

These limitations are stated explicitly and must not be hidden in the thesis or paper.

1. **Limited dataset diversity.** Training and primary testing use the Kermany
   pediatric chest X-ray dataset from a single institution. The supplementary
   RSNA experiment is one exploratory external probe, not multi-centre
   validation. Results may not transfer to other scanners, acquisition
   protocols, age groups, or hospitals.

2. **Limited external validation.** The model was evaluated zero-shot on one
   independent RSNA dataset using a simplified binary label mapping. This is
   supplementary evidence only; it does not replace multi-dataset,
   patient-level, or prospective clinical validation. NIH ChestX-ray14 and
   CheXpert were not evaluated.

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
- Static INT8 quantization is backend-dependent; if unsupported on the host, the
  pipeline reports this rather than substituting estimated numbers.
