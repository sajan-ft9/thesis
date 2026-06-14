"""Model factory: EfficientNet-B0 (primary) + ResNet-18 / MobileNetV3-Small baselines.

All three backbones are wrapped in :class:`PneumoniaNet`, which (a) replaces the
ImageNet classifier with a shared binary head, (b) exposes a uniform staged
freeze/unfreeze API for the transfer-learning schedule, and (c) reports the
correct Grad-CAM++ target layer per architecture. Using a single wrapper keeps
the training pipeline *identical* across models, which is what makes the baseline
comparison fair and publishable.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

from .utils import count_parameters

__all__ = ["PneumoniaNet", "build_model", "SUPPORTED_MODELS"]

SUPPORTED_MODELS: tuple[str, ...] = ("efficientnet_b0", "resnet18", "mobilenetv3_small")


def _make_head(in_features: int, hidden_dim: int, dropout_head: float, dropout_classifier: float) -> nn.Sequential:
    """Shared binary classification head (single logit output)."""
    head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout_head),
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_classifier),
        nn.Linear(hidden_dim, 1),
    )
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return head


class PneumoniaNet(nn.Module):
    """A torchvision backbone with a shared binary head and a uniform freeze API."""

    def __init__(
        self,
        name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_stages: int = 5,
        hidden_dim: int = 256,
        dropout_head: float = 0.3,
        dropout_classifier: float = 0.2,
    ) -> None:
        super().__init__()
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{name}'. Choose from {SUPPORTED_MODELS}.")
        self.name = name
        weights = "DEFAULT" if pretrained else None

        if name == "efficientnet_b0":
            self.net = tvm.efficientnet_b0(weights=weights)
            in_features = self.net.classifier[1].in_features
            self.net.classifier = _make_head(in_features, hidden_dim, dropout_head, dropout_classifier)
            self._stages = list(self.net.features.children())
            self._cam_layers = [self.net.features[-1]]
        elif name == "resnet18":
            self.net = tvm.resnet18(weights=weights)
            in_features = self.net.fc.in_features
            self.net.fc = _make_head(in_features, hidden_dim, dropout_head, dropout_classifier)
            self._stages = [
                self.net.conv1,
                self.net.bn1,
                self.net.layer1,
                self.net.layer2,
                self.net.layer3,
                self.net.layer4,
            ]
            self._cam_layers = [self.net.layer4[-1]]
        else:  # mobilenetv3_small
            self.net = tvm.mobilenet_v3_small(weights=weights)
            in_features = self.net.classifier[0].in_features
            self.net.classifier = _make_head(in_features, hidden_dim, dropout_head, dropout_classifier)
            self._stages = list(self.net.features.children())
            self._cam_layers = [self.net.features[-1]]

        self.freeze_stages(freeze_stages)

    def forward(self, x):  # noqa: D102 - standard forward
        return self.net(x)

    def freeze_stages(self, num_stages: int) -> None:
        """Freeze the first ``num_stages`` backbone stages (head stays trainable)."""
        for i, stage in enumerate(self._stages):
            requires_grad = i >= num_stages
            for param in stage.parameters():
                param.requires_grad = requires_grad

    def unfreeze_all(self) -> None:
        """Make every parameter trainable (for the fine-tuning phase)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_cam_target_layers(self) -> list[nn.Module]:
        """Return the convolutional layer(s) to target for Grad-CAM++."""
        return self._cam_layers


def build_model(
    name: str = "efficientnet_b0",
    pretrained: bool = True,
    freeze_stages: int = 5,
    hidden_dim: int = 256,
    dropout_head: float = 0.3,
    dropout_classifier: float = 0.2,
) -> PneumoniaNet:
    """Convenience factory mirroring the :class:`PneumoniaNet` constructor."""
    model = PneumoniaNet(
        name=name,
        pretrained=pretrained,
        freeze_stages=freeze_stages,
        hidden_dim=hidden_dim,
        dropout_head=dropout_head,
        dropout_classifier=dropout_classifier,
    )
    total, trainable = count_parameters(model)
    pct = 100.0 * trainable / total if total else 0.0
    from .utils import get_logger

    get_logger("models").info(
        "Built %s | total params=%s trainable=%s (%.1f%%)", name, f"{total:,}", f"{trainable:,}", pct
    )
    return model
