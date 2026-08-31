"""Dataset, splitting, validation and DataLoader construction.

The loader is *lazy and path-based*: images are read from disk one sample at a
time, so the full dataset never resides in RAM. This is the memory-safe design
central to the thesis and is what makes training feasible on constrained hardware.

The module also provides scientific-integrity tooling that the original notebook
lacked: class-balance reporting, content-hash **duplicate detection**, and an
explicit **train/val/test leakage check**.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .transforms import build_transforms
from .utils import get_logger, make_generator, seed_worker

__all__ = [
    "ChestXRayDataset",
    "DataBundle",
    "CLASS_NAMES",
    "scan_split",
    "class_distribution",
    "stratified_split",
    "find_duplicates",
    "verify_no_leakage",
    "compute_dataset_statistics",
    "build_dataloaders",
    "build_dataloaders_from_samples",
    "deduplicate_samples",
]

logger = get_logger("dataset")

CLASS_MAP: dict[str, int] = {"normal": 0, "pneumonia": 1}
CLASS_NAMES: tuple[str, str] = ("Normal", "Pneumonia")
IMAGE_EXTENSIONS: set[str] = {".jpeg", ".jpg", ".png", ".bmp"}

Sample = tuple[str, int]


def scan_split(split_dir: str | Path) -> list[Sample]:
    """Scan a ``<split>/{NORMAL,PNEUMONIA}`` directory into ``(path, label)`` pairs.

    Returns a *sorted* list for deterministic ordering. Raises ``FileNotFoundError``
    if the directory is missing and ``ValueError`` if no images are found.
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    samples: list[Sample] = []
    for class_name in sorted(os.listdir(split_dir)):
        label = CLASS_MAP.get(class_name.lower())
        class_dir = split_dir / class_name
        if label is None or not class_dir.is_dir():
            continue
        for fname in sorted(os.listdir(class_dir)):
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(class_dir / fname), label))
    if not samples:
        raise ValueError(f"No images found under {split_dir}")
    return sorted(samples)


class ChestXRayDataset(Dataset):
    """Lazy chest X-ray dataset over an explicit list of ``(path, label)`` pairs."""

    def __init__(self, samples: list[Sample], transform=None, name: str = "dataset") -> None:
        self.samples = list(samples)
        self.transform = transform
        self.name = name

    @classmethod
    def from_directory(cls, split_dir: str | Path, transform=None, name: str | None = None) -> ChestXRayDataset:
        """Build a dataset by scanning a split directory."""
        samples = scan_split(split_dir)
        return cls(samples, transform=transform, name=name or Path(split_dir).name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    @property
    def paths(self) -> list[str]:
        return [path for path, _ in self.samples]


def class_distribution(samples: Iterable[Sample]) -> dict[str, int]:
    """Return ``{class_name: count}`` for a collection of samples."""
    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in samples:
        counts[CLASS_NAMES[label]] += 1
    return counts


def stratified_split(
    samples: list[Sample], val_split: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Stratified train/validation split preserving class proportions."""
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1); got {val_split}")
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    p_train, p_val, y_train, y_val = train_test_split(
        paths, labels, test_size=val_split, stratify=labels, random_state=seed
    )
    train = sorted(zip(p_train, y_train))
    val = sorted(zip(p_val, y_val))
    return list(train), list(val)


def _file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file's bytes."""
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def deduplicate_samples(samples: Iterable[Sample]) -> tuple[list[Sample], list[str]]:
    """Remove byte-identical duplicate images, keeping the first by sorted path.

    Returns ``(kept_samples, removed_paths)``. Used to prevent the same image
    leaking across the train/validation split (a known quirk of the Kermany set).
    """
    seen: set[str] = set()
    kept: list[Sample] = []
    removed: list[str] = []
    for path, label in sorted(samples):
        digest = _file_md5(path)
        if digest in seen:
            removed.append(path)
        else:
            seen.add(digest)
            kept.append((path, label))
    return kept, removed


def find_duplicates(samples: Iterable[Sample]) -> dict[str, list[str]]:
    """Detect byte-identical duplicate images via MD5 hashing.

    Returns ``{md5: [paths...]}`` for every hash that maps to more than one file.
    """
    by_hash: dict[str, list[str]] = {}
    for path, _ in samples:
        digest = _file_md5(path)
        by_hash.setdefault(digest, []).append(path)
    return {digest: paths for digest, paths in by_hash.items() if len(paths) > 1}


def verify_no_leakage(
    splits: dict[str, list[Sample]], by_content: bool = True
) -> dict[str, object]:
    """Verify splits are disjoint by path and (optionally) by image content.

    Parameters
    ----------
    splits:
        Mapping of split name -> samples (e.g. ``{"train": ..., "val": ...}``).
    by_content:
        If True, also hash every image and flag the same content appearing in
        more than one split (catches duplicated files across splits).

    Returns
    -------
    dict
        ``{"path_leaks": {...}, "content_leaks": {...}, "clean": bool}``.
    """
    # Path-level overlap.
    path_sets = {name: {p for p, _ in s} for name, s in splits.items()}
    names = list(path_sets)
    path_leaks: dict[str, list[str]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = path_sets[names[i]] & path_sets[names[j]]
            if overlap:
                path_leaks[f"{names[i]}|{names[j]}"] = sorted(overlap)

    content_leaks: dict[str, list[str]] = {}
    if by_content:
        hash_to_splits: dict[str, set[str]] = {}
        for name, s in splits.items():
            for path, _ in s:
                digest = _file_md5(path)
                hash_to_splits.setdefault(digest, set()).add(name)
        for digest, owning in hash_to_splits.items():
            if len(owning) > 1:
                content_leaks[digest] = sorted(owning)

    clean = not path_leaks and not content_leaks
    return {"path_leaks": path_leaks, "content_leaks": content_leaks, "clean": clean}


def compute_dataset_statistics(cfg: Config, seed: int | None = None) -> dict[str, object]:
    """Compute per-split class counts and balance for the dataset statistics table."""
    seed = cfg.seed if seed is None else seed
    root = Path(cfg.data.root)
    train_samples = scan_split(root / cfg.data.train_dirname)
    test_samples = scan_split(root / cfg.data.test_dirname)
    train_only, val_only = stratified_split(train_samples, cfg.data.val_split, seed)

    rows = []
    for split_name, samples in [
        ("train", train_only),
        ("val", val_only),
        ("test", test_samples),
        ("total", train_samples + test_samples),
    ]:
        dist = class_distribution(samples)
        total = sum(dist.values())
        rows.append(
            {
                "split": split_name,
                "normal": dist["Normal"],
                "pneumonia": dist["Pneumonia"],
                "total": total,
                "pneumonia_ratio": round(dist["Pneumonia"] / total, 4) if total else 0.0,
            }
        )
    return {"per_split": rows, "seed": seed, "img_size": cfg.data.img_size}


@dataclass
class DataBundle:
    """Container bundling DataLoaders, datasets and the underlying sample lists."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: ChestXRayDataset
    val_dataset: ChestXRayDataset
    test_dataset: ChestXRayDataset
    train_samples: list[Sample]
    val_samples: list[Sample]
    test_samples: list[Sample]


def build_dataloaders_from_samples(
    cfg: Config,
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    seed: int,
    drop_last: bool = True,
) -> DataBundle:
    """Build a :class:`DataBundle` from explicit sample lists (no scanning/splitting).

    Shared by :func:`build_dataloaders` (which derives the lists via one stratified
    split) and :mod:`scripts.run_kfold_cv` (which derives them per cross-validation
    fold) so the loader-construction logic — transforms, seeding, worker init — is
    identical either way.
    """
    train_transform, eval_transform = build_transforms(cfg)
    train_ds = ChestXRayDataset(train_samples, train_transform, name="train")
    val_ds = ChestXRayDataset(val_samples, eval_transform, name="val")
    test_ds = ChestXRayDataset(test_samples, eval_transform, name="test")

    generator = make_generator(seed)
    common = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=drop_last, generator=generator, **common
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )


def build_dataloaders(cfg: Config, seed: int | None = None, drop_last: bool = True) -> DataBundle:
    """Build train/val/test loaders with deterministic seeding and a stratified split."""
    seed = cfg.seed if seed is None else seed
    root = Path(cfg.data.root)

    all_train = scan_split(root / cfg.data.train_dirname)
    test_samples = scan_split(root / cfg.data.test_dirname)
    if cfg.data.deduplicate:
        all_train, removed = deduplicate_samples(all_train)
        if removed:
            logger.info("Deduplicated train pool: removed %d byte-identical duplicate image(s)", len(removed))
    train_samples, val_samples = stratified_split(all_train, cfg.data.val_split, seed)

    logger.info(
        "Split sizes -> train=%d val=%d test=%d", len(train_samples), len(val_samples), len(test_samples)
    )
    return build_dataloaders_from_samples(cfg, train_samples, val_samples, test_samples, seed, drop_last)


def validate_dataset(cfg: Config, check_duplicates: bool = True, check_leakage: bool = True) -> dict[str, object]:
    """Run integrity checks and return a validation report.

    Reports per-split class distribution, byte-identical duplicates within each
    split, and train/val/test leakage (by path and content hash). Intended to be
    run once before training and saved as a JSON artifact.
    """
    seed = cfg.seed
    root = Path(cfg.data.root)
    all_train_raw = scan_split(root / cfg.data.train_dirname)
    test_samples = scan_split(root / cfg.data.test_dirname)

    # Report the RAW (naive-split) state first, to expose the dataset's duplicate quirk.
    raw_train, raw_val = stratified_split(all_train_raw, cfg.data.val_split, seed)
    raw_splits = {"train": raw_train, "val": raw_val, "test": test_samples}

    # Then apply deduplication (as training does) and re-check.
    all_train, removed = (deduplicate_samples(all_train_raw) if cfg.data.deduplicate else (all_train_raw, []))
    train_samples, val_samples = stratified_split(all_train, cfg.data.val_split, seed)
    splits = {"train": train_samples, "val": val_samples, "test": test_samples}

    report: dict[str, object] = {
        "root": str(root),
        "seed": seed,
        "deduplicate": cfg.data.deduplicate,
        "n_train_duplicates_removed": len(removed),
        "class_distribution": {name: class_distribution(s) for name, s in splits.items()},
        "split_sizes": {name: len(s) for name, s in splits.items()},
        "raw_split_sizes": {name: len(s) for name, s in raw_splits.items()},
    }
    if check_duplicates:
        # Duplicates intrinsic to the raw splits (the dataset quirk we disclose).
        report["duplicates"] = {name: find_duplicates(s) for name, s in raw_splits.items()}
        report["n_duplicate_groups"] = {name: len(report["duplicates"][name]) for name in raw_splits}  # type: ignore[index]
    if check_leakage:
        # Leakage before dedup (problem detected) vs after dedup (what training uses).
        report["raw_leakage"] = verify_no_leakage(raw_splits, by_content=True)
        report["leakage"] = verify_no_leakage(splits, by_content=True)
    return report


def main() -> None:
    """CLI: validate the dataset and save a report + statistics table."""
    import argparse

    from .config import add_config_cli_args, config_from_cli
    from .utils import save_json

    parser = argparse.ArgumentParser(description="Validate dataset integrity and export statistics.")
    add_config_cli_args(parser)
    parser.add_argument("--no-duplicates", action="store_true", help="Skip duplicate detection (faster).")
    parser.add_argument("--no-leakage", action="store_true", help="Skip leakage check (faster).")
    args = parser.parse_args()
    cfg = config_from_cli(args)

    report = validate_dataset(cfg, check_duplicates=not args.no_duplicates, check_leakage=not args.no_leakage)
    out = Path(cfg.paths.results_dir) / "metrics" / "dataset_validation.json"
    save_json(report, out)

    leak = report.get("leakage", {})
    logger.info("Split sizes: %s", report["split_sizes"])
    logger.info("Class distribution: %s", report["class_distribution"])
    if isinstance(leak, dict):
        logger.info("Leakage clean: %s", leak.get("clean"))
    logger.info("Saved validation report to %s", out)


if __name__ == "__main__":
    main()
