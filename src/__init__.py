"""Memory-Efficient, Explainable, Quantized Pneumonia Detection.

A reproducible research package for chest X-ray pneumonia classification with
EfficientNet-B0 (plus ResNet-18 / MobileNetV3-Small baselines), Grad-CAM++
explainability, INT8 quantization, and rigorous efficiency benchmarking.

The package is fully script-driven: every stage exposes a command-line entry
point (``python -m src.<module>``) and is configured via YAML files in
``configs/``. No experimental logic lives in notebooks.
"""

__version__ = "1.0.0"
