"""Grad-CAM++ visual explanations.

Generates overlay heatmaps for three diagnostically meaningful categories drawn
from the *real* test-set predictions: correctly classified cases, false
positives, and false negatives. This supports qualitative error analysis and
clinical interpretability discussion.

IMPORTANT (scientific integrity): this module produces only model-derived
visualisations. It does **not** simulate radiologist ratings, trust scores, or
any clinical validation — those require an IRB-approved human study and are
documented as future work, never fabricated.

Run:
    python -m src.explainability --checkpoint models/efficientnet_b0_best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .config import Config  # noqa: E402
from .dataset import CLASS_NAMES, build_dataloaders  # noqa: E402
from .inference import collect_predictions, load_checkpoint  # noqa: E402
from .utils import ensure_dir, get_device, get_logger, save_json  # noqa: E402

logger = get_logger("explain")

__all__ = ["denormalize", "generate_overlay", "run_explainability"]


def denormalize(tensor: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """Undo normalisation and return an HWC float image in [0, 1]."""
    mean_arr = np.asarray(mean)
    std_arr = np.asarray(std)
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std_arr + mean_arr
    return np.clip(img, 0, 1).astype(np.float32)


def generate_overlay(cam_engine, model, img_tensor: torch.Tensor, cfg: Config, device: torch.device):
    """Return ``(original_rgb, overlay_rgb, pred_prob, pred_class)`` for one image."""
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    input_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(torch.sigmoid(model(input_tensor).squeeze()).item())
    pred_class = int(prob >= cfg.evaluate.threshold)

    grayscale = cam_engine(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(pred_class)])[0]
    original = denormalize(img_tensor, cfg.transforms.norm_mean, cfg.transforms.norm_std)
    overlay = show_cam_on_image(original, grayscale, use_rgb=True).astype(np.float32) / 255.0
    return original, overlay, prob, pred_class


def _select_indices(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> dict[str, list[int]]:
    correct = np.where(y_pred == y_true)[0]
    fp = np.where((y_pred == 1) & (y_true == 0))[0]
    fn = np.where((y_pred == 0) & (y_true == 1))[0]
    return {
        "correct": correct[:n].tolist(),
        "false_positive": fp[:n].tolist(),
        "false_negative": fn[:n].tolist(),
    }


def run_explainability(checkpoint_path: str | Path, cfg: Config | None = None, device_str: str = "auto") -> dict[str, Any]:
    """Generate and save Grad-CAM++ overlays for correct / FP / FN test cases."""
    from pytorch_grad_cam import GradCAMPlusPlus

    device = get_device(device_str)
    model, ckpt_cfg = load_checkpoint(checkpoint_path, device)
    cfg = cfg or ckpt_cfg
    data = build_dataloaders(cfg, seed=cfg.seed)

    y_true, y_prob, y_pred = collect_predictions(model, data.test_loader, device, cfg.evaluate.threshold)
    selection = _select_indices(y_true, y_pred, cfg.explain.num_per_category)

    out_dir = ensure_dir(Path(cfg.paths.results_dir) / "gradcam")
    cam_engine = GradCAMPlusPlus(model=model, target_layers=model.get_cam_target_layers())

    records: list[dict[str, Any]] = []
    grid_items: list[tuple[np.ndarray, np.ndarray, str]] = []
    for category, indices in selection.items():
        for idx in indices:
            img_tensor, label = data.test_dataset[idx]
            original, overlay, prob, pred_class = generate_overlay(cam_engine, model, img_tensor, cfg, device)
            true_str = CLASS_NAMES[int(label.item())]
            pred_str = CLASS_NAMES[pred_class]
            fname = f"{cfg.experiment_name}_{category}_{idx:04d}_true{true_str}_pred{pred_str}_p{prob:.2f}.png"
            plt.imsave(str(out_dir / fname), overlay)
            records.append(
                {
                    "category": category,
                    "test_index": int(idx),
                    "true_label": true_str,
                    "pred_label": pred_str,
                    "probability": round(prob, 4),
                    "filename": fname,
                }
            )
            if len(grid_items) < 6:
                grid_items.append((original, overlay, f"{category}\nT:{true_str} P:{pred_str} ({prob:.2f})"))

    records_df = pd.DataFrame(records)
    records_csv = Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_gradcam_records.csv"
    ensure_dir(records_csv.parent)
    records_df.to_csv(records_csv, index=False)

    _save_grid(grid_items, Path(cfg.paths.results_dir) / "figures", f"{cfg.experiment_name}_gradcam_grid")

    summary = {
        "experiment_name": cfg.experiment_name,
        "n_overlays": len(records),
        "by_category": {k: len(v) for k, v in selection.items()},
        "records_csv": str(records_csv),
        "gradcam_dir": str(out_dir),
    }
    save_json(summary, Path(cfg.paths.results_dir) / "metrics" / f"{cfg.experiment_name}_explainability.json")
    logger.info("Saved %d Grad-CAM++ overlays to %s", len(records), out_dir)
    return summary


def _save_grid(items: list[tuple[np.ndarray, np.ndarray, str]], out_dir: str | Path, stem: str) -> None:
    """Save a 2-row grid (original CXR / Grad-CAM++ overlay) for a few examples."""
    if not items:
        return
    out_dir = ensure_dir(out_dir)
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, (original, overlay, title) in enumerate(items):
        axes[0, col].imshow(original)
        axes[0, col].set_title(title, fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM++", fontsize=9)
    fig.suptitle("Grad-CAM++ Visual Explanations", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ explanations for a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_explainability(args.checkpoint, device_str=args.device)


if __name__ == "__main__":
    main()
