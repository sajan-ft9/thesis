"""Inference utilities and CLI.

Provides checkpoint loading, batched prediction collection (shared by evaluation
and quantization), and single-image / folder inference from the command line.

Run:
    python -m src.inference --checkpoint models/efficientnet_b0_best.pth --image path/to/cxr.jpeg
    python -m src.inference --checkpoint models/efficientnet_b0_best.pth --folder path/to/images/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import Config
from .models import PneumoniaNet
from .transforms import build_eval_transform
from .utils import get_device, get_logger

logger = get_logger("inference")

__all__ = ["load_checkpoint", "collect_predictions", "predict_image", "IMAGE_EXTENSIONS"]

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


def load_checkpoint(
    checkpoint_path: str | Path, device: torch.device | None = None, pretrained_backbone: bool = False
) -> tuple[PneumoniaNet, Config]:
    """Load a trained model + its config from a checkpoint saved by ``train.py``."""
    device = device or torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = PneumoniaNet(
        name=cfg.model.name,
        pretrained=pretrained_backbone,  # weights come from the checkpoint, not ImageNet
        freeze_stages=0,
        hidden_dim=cfg.model.hidden_dim,
        dropout_head=cfg.model.dropout_head,
        dropout_classifier=cfg.model.dropout_classifier,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    logger.info("Loaded %s checkpoint from %s (best epoch %s)", cfg.model.name, checkpoint_path, ckpt.get("epoch"))
    return model, cfg


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader, device: torch.device, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a loader through a model; return ``(y_true, y_prob, y_pred)`` as arrays."""
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for imgs, batch_labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).squeeze(1)
        batch_probs = torch.sigmoid(logits).float().cpu().numpy()
        probs.extend(batch_probs.tolist())
        labels.extend(batch_labels.cpu().numpy().astype(int).tolist())
    y_prob = np.asarray(probs, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_prob, y_pred


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    cfg: Config,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict pneumonia probability for a single image file."""
    transform = build_eval_transform(cfg)
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    prob = float(torch.sigmoid(model(tensor).squeeze()).item())
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    return {"path": str(image_path), "probability": prob, "prediction": label}


def _iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a *_best.pth checkpoint.")
    parser.add_argument("--image", help="Path to a single image.")
    parser.add_argument("--folder", help="Path to a folder of images (recursive).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder.")

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)

    targets = [Path(args.image)] if args.image else list(_iter_images(Path(args.folder)))
    for path in targets:
        result = predict_image(model, path, cfg, device, args.threshold)
        print(f"{result['prediction']:<10} p={result['probability']:.4f}  {result['path']}")


if __name__ == "__main__":
    main()
