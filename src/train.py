"""Training entry point.

Implements the staged transfer-learning schedule (freeze head-only warmup, then
full fine-tune), mixed-precision training, gradient clipping, cosine LR with
linear warmup, early stopping on validation AUC, checkpointing of the best model,
and full metric logging to CSV/JSON (+ optional TensorBoard). A reproducibility
manifest (``metadata.json``) is written alongside every run.

Run:
    python -m src.train --config configs/efficientnet_b0.yaml
    python -m src.train --config configs/resnet18.yaml --override train.epochs=1
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from .config import Config, add_config_cli_args, config_from_cli
from .dataset import build_dataloaders, class_distribution
from .metrics import compute_metrics
from .models import build_model
from .utils import build_metadata, ensure_dir, get_device, get_logger, save_json, seed_everything

logger = get_logger("train")


class SmoothBCELoss(nn.Module):
    """Binary cross-entropy with label smoothing on logits.

    Targets in {0, 1} are softened towards 0.5 by ``smoothing``, which regularises
    the classifier and reduces over-confidence — useful given noisy CXR labels.
    """

    def __init__(self, smoothing: float = 0.05) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def build_scheduler(optimizer: optim.Optimizer, cfg: Config) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing to ``lr_min`` (epoch-stepped)."""
    warmup = max(cfg.train.warmup_epochs, 1)
    total = max(cfg.train.epochs, warmup + 1)
    floor = cfg.train.lr_min / cfg.train.lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        return max(floor, 0.5 * (1 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    grad_clip: float = 1.0,
    amp_enabled: bool = False,
    training: bool = True,
) -> tuple[float, dict[str, float], list[int], list[float]]:
    """Run a single epoch; returns ``(avg_loss, metrics, y_true, y_prob)``."""
    model.train(training)
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []
    device_type = "cuda" if device.type == "cuda" else "cpu"

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, enabled=amp_enabled):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
            if training and optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics, all_labels, all_probs


def _monitor_value(metrics: dict[str, float], loss: float, monitor: str) -> float:
    """Extract the checkpoint-selection score (higher is better)."""
    if monitor == "val_loss":
        return -loss
    key = monitor.replace("val_", "")
    value = metrics.get(key, float("nan"))
    return -math.inf if (value is None or math.isnan(value)) else value


def train(cfg: Config) -> dict[str, Any]:
    """Full training run for a single model configuration. Returns a summary dict."""
    seed = seed_everything(cfg.seed)
    device = get_device(cfg.device)
    logger.info("Experiment '%s' | device=%s | seed=%d", cfg.experiment_name, device, seed)

    data = build_dataloaders(cfg, seed=seed)
    logger.info("Train class balance: %s", class_distribution(data.train_samples))

    model = build_model(
        name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        freeze_stages=cfg.model.freeze_stages,
        hidden_dim=cfg.model.hidden_dim,
        dropout_head=cfg.model.dropout_head,
        dropout_classifier=cfg.model.dropout_classifier,
    ).to(device)

    criterion = SmoothBCELoss(cfg.train.label_smoothing)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    amp_enabled = cfg.train.amp and device.type == "cuda"
    scaler = GradScaler(device="cuda", enabled=amp_enabled)
    scheduler = build_scheduler(optimizer, cfg)

    models_dir = ensure_dir(cfg.paths.models_dir)
    metrics_dir = ensure_dir(Path(cfg.paths.results_dir) / "metrics")
    ckpt_path = models_dir / f"{cfg.experiment_name}_best.pth"

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        if epoch == cfg.train.unfreeze_epoch:
            model.unfreeze_all()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.train.lr * cfg.train.unfreeze_lr_scale,
                weight_decay=cfg.train.weight_decay,
            )
            scheduler = build_scheduler(optimizer, cfg)
            logger.info("[epoch %d] backbone unfrozen; lr reset", epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_m, _, _ = run_epoch(
            model, data.train_loader, criterion, device, optimizer, scaler,
            cfg.train.grad_clip, amp_enabled, training=True,
        )
        va_loss, va_m, _, _ = run_epoch(
            model, data.val_loader, criterion, device, training=False,
        )
        scheduler.step()

        row = {
            "epoch": epoch, "lr": lr_now,
            "train_loss": tr_loss, "train_acc": tr_m["accuracy"], "train_auc": tr_m["auc"],
            "val_loss": va_loss, "val_acc": va_m["accuracy"], "val_auc": va_m["auc"],
            "val_precision": va_m["precision"], "val_recall": va_m["recall"], "val_f1": va_m["f1"],
        }
        history.append(row)

        score = _monitor_value(va_m, va_loss, cfg.train.monitor)
        note = ""
        if score > best_score:
            best_score, best_epoch, no_improve = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": cfg.model.name,
                    "state_dict": best_state,
                    "val_metrics": va_m,
                    "config": cfg.to_dict(),
                },
                ckpt_path,
            )
            note = "*best*"
        else:
            no_improve += 1

        logger.info(
            "epoch %02d | lr %.2e | tr_loss %.4f tr_auc %.4f | va_loss %.4f va_auc %.4f %s",
            epoch, lr_now, tr_loss, tr_m["auc"], va_loss, va_m["auc"], note,
        )
        if no_improve >= cfg.train.early_stopping_patience:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, no_improve)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist history + metadata.
    hist_df = pd.DataFrame(history)
    hist_csv = metrics_dir / f"{cfg.experiment_name}_history.csv"
    hist_df.to_csv(hist_csv, index=False)
    save_json(history, metrics_dir / f"{cfg.experiment_name}_history.json")

    dataset_stats = {
        "train": class_distribution(data.train_samples),
        "val": class_distribution(data.val_samples),
        "test": class_distribution(data.test_samples),
    }
    metadata = build_metadata(
        experiment_name=cfg.experiment_name,
        config=cfg.to_dict(),
        seed=seed,
        extra={
            "best_epoch": best_epoch,
            "best_val_score": best_score,
            "monitor": cfg.train.monitor,
            "dataset_stats": dataset_stats,
            "checkpoint": str(ckpt_path),
            "device": str(device),
        },
    )
    save_json(metadata, metrics_dir / f"{cfg.experiment_name}_metadata.json")

    _maybe_tensorboard(cfg, history)
    logger.info("Training complete. Best epoch=%d, checkpoint=%s", best_epoch, ckpt_path)
    return {"checkpoint": str(ckpt_path), "best_epoch": best_epoch, "history": history}


def _maybe_tensorboard(cfg: Config, history: list[dict[str, Any]]) -> None:
    """Write scalar curves to TensorBoard if the package is available (optional)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.info("TensorBoard not installed; skipping scalar logging.")
        return
    log_dir = ensure_dir(Path(cfg.paths.results_dir) / "tensorboard" / cfg.experiment_name)
    writer = SummaryWriter(log_dir=str(log_dir))
    for row in history:
        step = int(row["epoch"])
        for key, value in row.items():
            if key == "epoch" or value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            writer.add_scalar(key, value, step)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pneumonia detection model.")
    add_config_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_cli(args)
    train(cfg)


if __name__ == "__main__":
    main()
