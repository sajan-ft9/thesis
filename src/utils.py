"""Reproducibility, device, logging, and experiment-metadata utilities.

These helpers are deliberately framework-light so they can be unit-tested without
a GPU. The central guarantee is :func:`seed_everything`, which seeds Python,
NumPy and PyTorch *and* returns the pieces needed to make a ``DataLoader``
deterministic (a seeded ``Generator`` and a ``worker_init_fn``).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "seed_everything",
    "seed_worker",
    "make_generator",
    "get_device",
    "count_parameters",
    "get_git_sha",
    "library_versions",
    "build_metadata",
    "save_json",
    "load_json",
    "ensure_dir",
    "get_logger",
    "utc_timestamp",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "pneumonia", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide configured logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp (seconds precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all relevant RNGs for reproducible experiments.

    Seeds ``random``, ``numpy`` and ``torch`` (CPU + all CUDA devices), sets the
    ``PYTHONHASHSEED`` environment variable, and (optionally) forces cuDNN into
    deterministic mode. Returns the seed for convenience/logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch optional for pure-utility tests
        pass
    return seed


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` that makes each DataLoader worker deterministic.

    PyTorch seeds each worker's base RNG from the main seed; we propagate that to
    Python's ``random`` and to NumPy so augmentation pipelines are reproducible.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42):
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device(preference: str = "auto"):
    """Resolve a device string to a ``torch.device``.

    ``auto`` picks CUDA, then Apple MPS, then CPU. Explicit values are honoured
    but fall back to CPU with a warning if unavailable.
    """
    import torch

    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        get_logger().warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if pref == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        get_logger().warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def count_parameters(model) -> tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)`` for a module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_git_sha(short: bool = True) -> str | None:
    """Return the current git commit SHA, or ``None`` if not in a git repo."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            cmd = ["git", "rev-parse", "HEAD"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        sha = out.stdout.strip()
        return sha or None
    except (FileNotFoundError, OSError):
        return None


def library_versions() -> dict[str, str]:
    """Capture versions of the key scientific libraries for the manifest."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod_name in ("torch", "torchvision", "numpy", "sklearn", "timm"):
        try:
            module = __import__(mod_name)
            versions[mod_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not-installed"
    return versions


def build_metadata(
    *,
    experiment_name: str,
    config: dict[str, Any],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a reproducibility manifest dict (write with :func:`save_json`)."""
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp_utc": utc_timestamp(),
        "git_sha": get_git_sha(),
        "seed": seed,
        "library_versions": library_versions(),
        "config": config,
    }
    if extra:
        metadata.update(extra)
    return metadata


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed; return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> Path:
    """Serialise ``obj`` to pretty JSON, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return p


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for NumPy scalars/arrays and Paths."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
