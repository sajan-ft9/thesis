# Future Work

These directions extend naturally into a PhD research programme on trustworthy,
resource-efficient medical AI.

1. **External validation on NIH ChestX-ray14.** Assess generalisation to a large,
   adult, multi-pathology dataset and quantify domain shift.

2. **Cross-dataset validation on CheXpert.** Evaluate against expert-labelled,
   uncertainty-aware annotations and study calibration under distribution shift.

3. **Edge-hardware deployment.** Benchmark the quantized model on Raspberry Pi 4
   and NVIDIA Jetson Nano for real latency, memory, energy and thermal behaviour.

4. **TinyML optimisation.** Explore quantization-aware training, structured
   pruning, and knowledge distillation to push below microcontroller-class budgets.

5. **Federated learning.** Train across institutions without sharing raw images to
   improve generalisation while preserving patient privacy.

6. **Multi-disease chest X-ray diagnosis.** Extend from binary pneumonia detection
   to multi-label thoracic disease classification with calibrated, explainable
   per-finding outputs.
