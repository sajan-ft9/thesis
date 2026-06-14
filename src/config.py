"""Typed, hierarchical configuration loading.

Configurations are YAML files that may inherit from a parent via an ``extends``
key (resolved relative to the child file). Values can be overridden on the
command line with ``--override dotted.key=value`` (parsed as YAML, so types are
preserved). The merged dictionary is materialised into nested dataclasses for
attribute access and static checking.

Example
-------
>>> cfg = load_config("configs/efficientnet_b0.yaml",
...                    overrides=["train.epochs=1", "data.batch_size=8"])
>>> cfg.model.name
'efficientnet_b0'
>>> cfg.train.epochs
1
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "DataConfig",
    "TransformConfig",
    "ModelConfig",
    "TrainConfig",
    "EvaluateConfig",
    "BenchmarkConfig",
    "QuantizeConfig",
    "ExplainConfig",
    "PathsConfig",
    "load_config",
    "add_config_cli_args",
    "config_from_cli",
]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    root: str = "data/raw/chest_xray"
    train_dirname: str = "train"
    test_dirname: str = "test"
    val_split: float = 0.20
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class TransformConfig:
    norm_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    norm_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    train: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_stages: int = 5
    hidden_dim: int = 256
    dropout_head: float = 0.3
    dropout_classifier: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 3e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    unfreeze_epoch: int = 8
    unfreeze_lr_scale: float = 0.1
    label_smoothing: float = 0.05
    early_stopping_patience: int = 7
    grad_clip: float = 1.0
    amp: bool = True
    monitor: str = "val_auc"


@dataclass
class EvaluateConfig:
    bootstrap_n: int = 1000
    bootstrap_alpha: float = 0.05
    threshold: float = 0.5


@dataclass
class BenchmarkConfig:
    warmup: int = 10
    repeats: int = 50
    throughput_batch_size: int = 32
    device: str = "cpu"


@dataclass
class QuantizeConfig:
    backend: str = "auto"
    calibration_batches: int = 16


@dataclass
class ExplainConfig:
    method: str = "gradcampp"
    num_per_category: int = 4


@dataclass
class PathsConfig:
    models_dir: str = "models"
    results_dir: str = "results"


@dataclass
class Config:
    """Top-level configuration object."""

    seed: int = 42
    device: str = "auto"
    experiment_name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # -- (de)serialisation ------------------------------------------------- #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a :class:`Config` from a (possibly partial) nested dict."""
        data = copy.deepcopy(data)
        section_types: dict[str, type] = {
            "data": DataConfig,
            "transforms": TransformConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "evaluate": EvaluateConfig,
            "benchmark": BenchmarkConfig,
            "quantize": QuantizeConfig,
            "explain": ExplainConfig,
            "paths": PathsConfig,
        }
        kwargs: dict[str, Any] = {}
        for key in ("seed", "device", "experiment_name"):
            if key in data:
                kwargs[key] = data[key]
        for name, dc_type in section_types.items():
            section = data.get(name, {}) or {}
            valid = {f.name for f in dc_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            unknown = set(section) - valid
            if unknown:
                raise ValueError(f"Unknown keys in config section '{name}': {sorted(unknown)}")
            kwargs[name] = dc_type(**section)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON/YAML-serialisable nested dict."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``extends`` parent chain."""
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular 'extends' detected involving {path}")
    _seen.add(path)

    raw = _load_yaml(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    parent_path = (path.parent / parent_ref).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent config '{parent_ref}' referenced by {path} not found.")
    parent = _resolve_extends(parent_path, _seen)
    return _deep_merge(parent, raw)


def _apply_override(data: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply a single ``a.b.c=value`` override in place (value parsed as YAML)."""
    keys = dotted_key.split(".")
    value = yaml.safe_load(raw_value)
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override '{dotted_key}': '{key}' is not a mapping.")
    node[keys[-1]] = value


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a config file (resolving ``extends``) and apply CLI overrides.

    Parameters
    ----------
    config_path:
        Path to a YAML config file.
    overrides:
        Optional list of ``dotted.key=value`` strings; values are parsed as YAML.

    Returns
    -------
    Config
        The fully materialised, validated configuration object.
    """
    merged = _resolve_extends(Path(config_path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.subkey=value")
        key, value = item.split("=", 1)
        _apply_override(merged, key.strip(), value.strip())
    return Config.from_dict(merged)


# --------------------------------------------------------------------------- #
# argparse integration
# --------------------------------------------------------------------------- #
def add_config_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the standard ``--config`` / ``--override`` arguments to a parser."""
    parser.add_argument(
        "--config",
        type=str,
        default="configs/efficientnet_b0.yaml",
        help="Path to a YAML config file (default: configs/efficientnet_b0.yaml).",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        metavar="key.sub=value",
        help="Override config entries, e.g. --override train.epochs=1 data.batch_size=8",
    )
    return parser


def config_from_cli(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from parsed argparse args (``config`` + ``override``)."""
    return load_config(args.config, overrides=args.override)
