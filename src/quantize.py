"""Post-training quantization (dynamic INT8 + static PTQ) and ONNX export.

Two quantization schemes are compared honestly:

* **Dynamic INT8** — quantizes ``nn.Linear`` weights only; activations quantized
  per-batch at runtime. Trivial to apply but yields modest compression on a
  convolution-heavy backbone (most parameters live in conv layers).
* **Static PTQ** — FX graph-mode post-training static quantization with a
  calibration pass; quantizes convolutions too, giving the deployment-relevant
  size/latency reduction. Backend-dependent, so failures are reported, never
  faked.

For each variant we report model size, accuracy and ROC-AUC *before vs after*,
so the accuracy/efficiency trade-off is transparent.

Run:
    python -m src.quantize --checkpoint models/efficientnet_b0_best.pth
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .benchmarking import benchmark_latency, file_size_mb
from .config import Config
from .dataset import build_dataloaders
from .inference import collect_predictions, load_checkpoint
from .metrics import compute_metrics
from .utils import ensure_dir, get_logger, save_json

logger = get_logger("quantize")

__all__ = ["pick_backend", "dynamic_quantize", "static_quantize", "export_onnx", "run_quantization_study"]


def pick_backend(preference: str = "auto") -> str:
    """Choose a quantization backend supported on this platform."""
    supported = list(getattr(torch.backends.quantized, "supported_engines", []))
    if preference != "auto" and preference in supported:
        return preference
    for candidate in ("qnnpack", "x86", "fbgemm", "onednn"):
        if candidate in supported:
            return candidate
    return supported[0] if supported else "qnnpack"


def dynamic_quantize(model: nn.Module, backend: str = "auto") -> nn.Module:
    """Apply dynamic INT8 quantization to Linear layers (CPU)."""
    torch.backends.quantized.engine = pick_backend(backend)
    model_cpu = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)


def static_quantize(
    model: nn.Module, calib_loader, backend: str, img_size: int = 224, calibration_batches: int = 16
) -> nn.Module:
    """FX graph-mode static post-training quantization with calibration (CPU).

    Uses **per-channel symmetric INT8 weights + histogram activation observers**,
    which is essential for convolution-heavy / SiLU networks like EfficientNet-B0:
    the backend *default* (per-tensor) qconfig catastrophically degrades them
    (AUC collapses below 0.6). Calibrate on a CLEAN (non-augmented) loader — never
    the training loader with augmentation, and never the test set.

    Raises on tracing/backend failure; the caller is expected to catch and report
    the failure honestly rather than substitute fabricated numbers.
    """
    from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig_mapping
    from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = backend
    example_inputs = (torch.randn(1, 3, img_size, img_size),)

    def _build(mapping: QConfigMapping) -> nn.Module:
        model_cpu = copy.deepcopy(model).cpu().eval()
        prepared = prepare_fx(model_cpu, mapping, example_inputs)
        with torch.no_grad():
            for i, (imgs, _) in enumerate(calib_loader):
                prepared(imgs)
                if i + 1 >= calibration_batches:
                    break
        return convert_fx(prepared)

    per_channel = QConfig(
        activation=HistogramObserver.with_args(reduce_range=False),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    )
    try:
        return _build(QConfigMapping().set_global(per_channel))
    except Exception as exc:  # noqa: BLE001 - platform may not support per-channel; report and fall back
        logger.warning("Per-channel static PTQ failed (%s); falling back to backend default qconfig.", exc)
        return _build(get_default_qconfig_mapping(backend))


def export_onnx(model: nn.Module, path: str | Path, img_size: int = 224, opset: int = 13) -> bool:
    """Export an FP32 model to ONNX and verify it loads in onnxruntime."""
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    export_kwargs = {
        "input_names": ["chest_xray"],
        "output_names": ["pneumonia_logit"],
        "dynamic_axes": {"chest_xray": {0: "batch"}, "pneumonia_logit": {0: "batch"}},
        "opset_version": opset,
        "do_constant_folding": True,
    }
    try:
        try:
            # Force the stable TorchScript exporter (avoids the onnxscript dependency
            # required by the newer dynamo-based exporter in recent torch versions).
            torch.onnx.export(model, dummy, str(path), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(path), **export_kwargs)
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})
        logger.info("ONNX export verified at %s", path)
        return True
    except Exception as exc:  # noqa: BLE001 - report, do not crash the pipeline
        logger.warning("ONNX export failed: %s", exc)
        return False


def _evaluate_cpu(model: nn.Module, loader, threshold: float) -> dict[str, float]:
    y_true, y_prob, _ = collect_predictions(model, loader, torch.device("cpu"), threshold)
    return compute_metrics(y_true, y_prob, threshold)


def run_quantization_study(checkpoint_path: str | Path, cfg: Config | None = None) -> dict[str, Any]:
    """Compare FP32 vs dynamic-INT8 vs static-PTQ on size, latency and accuracy."""
    device = torch.device("cpu")  # quantized inference is CPU-only
    model_fp32, ckpt_cfg = load_checkpoint(checkpoint_path, device)
    cfg = cfg or ckpt_cfg
    data = build_dataloaders(cfg, seed=cfg.seed)
    img = cfg.data.img_size
    threshold = cfg.evaluate.threshold
    backend = pick_backend(cfg.quantize.backend)
    logger.info("Quantization backend: %s", backend)

    models_dir = ensure_dir(cfg.paths.models_dir)
    variants: list[dict[str, Any]] = []

    def record(model: nn.Module, label: str, artifact_path: Path) -> dict[str, Any]:
        # Size is measured from the actual serialized file that would be deployed
        # (not an in-memory re-serialization), so e.g. TorchScript archive overhead
        # for the static-PTQ variant is reflected honestly.
        metrics = _evaluate_cpu(model, data.test_loader, threshold)
        latency = benchmark_latency(model, device, (1, 3, img, img), cfg.benchmark.warmup, cfg.benchmark.repeats)
        row = {
            "variant": label,
            "size_mb": round(file_size_mb(artifact_path), 3),
            "accuracy": round(metrics["accuracy"], 4),
            "auc": round(metrics["auc"], 4),
            "sensitivity": round(metrics["sensitivity"], 4),
            "specificity": round(metrics["specificity"], 4),
            "latency_ms_mean": round(latency.mean_ms, 3),
            "latency_ms_std": round(latency.std_ms, 3),
        }
        variants.append(row)
        return row

    # FP32 baseline.
    fp32_path = models_dir / f"{cfg.experiment_name}_fp32.pth"
    torch.save(model_fp32.state_dict(), fp32_path)
    fp32_row = record(model_fp32, "FP32", fp32_path)

    # Dynamic INT8.
    try:
        dyn = dynamic_quantize(model_fp32, cfg.quantize.backend)
        dyn_path = models_dir / f"{cfg.experiment_name}_int8_dynamic.pth"
        torch.save(dyn.state_dict(), dyn_path)
        record(dyn, "INT8 dynamic", dyn_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dynamic quantization failed: %s", exc)

    # Static PTQ.
    static_supported = True
    try:
        # Calibrate on the clean (non-augmented) validation loader; never the test set.
        stat = static_quantize(model_fp32, data.val_loader, backend, img, cfg.quantize.calibration_batches)
        static_path = models_dir / f"{cfg.experiment_name}_int8_static.pt"
        try:
            torch.jit.save(torch.jit.script(stat), str(static_path))
        except Exception:  # noqa: BLE001 - scripting is best-effort
            static_path = models_dir / f"{cfg.experiment_name}_int8_static.pth"
            torch.save(stat.state_dict(), static_path)
        record(stat, "INT8 static (PTQ)", static_path)
    except Exception as exc:  # noqa: BLE001
        static_supported = False
        logger.warning("Static PTQ unsupported/failed on backend '%s': %s", backend, exc)

    # ONNX (FP32 deployment artifact).
    onnx_ok = export_onnx(model_fp32, models_dir / f"{cfg.experiment_name}.onnx", img)

    # Drops relative to FP32.
    for row in variants:
        row["acc_drop_pct"] = round((fp32_row["accuracy"] - row["accuracy"]) * 100, 3)
        row["auc_drop_pct"] = round((fp32_row["auc"] - row["auc"]) * 100, 3)
        row["size_reduction_pct"] = round((1 - row["size_mb"] / fp32_row["size_mb"]) * 100, 1)

    summary = {
        "experiment_name": cfg.experiment_name,
        "backend": backend,
        "static_ptq_supported": static_supported,
        "onnx_export_ok": onnx_ok,
        "variants": variants,
    }
    out = Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_quantization.json"
    save_json(summary, out)

    from .visualization import plot_quantization_comparison

    plot_quantization_comparison(variants, Path(cfg.paths.results_dir) / "figures")
    logger.info("Quantization study saved to %s", out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the quantization study on a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    run_quantization_study(args.checkpoint)


if __name__ == "__main__":
    main()
