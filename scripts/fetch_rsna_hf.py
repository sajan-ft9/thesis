"""Download the RSNA Pneumonia Detection Challenge data (DICOM) for EXTERNAL validation.

Uses the credential-free HuggingFace mirror ``Baldezo313/rsna-pneumonia-dataset`` (the
labels are the expert-adjudicated RSNA challenge labels). Downloads the train DICOMs and
the label CSVs into ``data/raw/rsna/``. Pair with ``scripts/build_rsna_subset.py`` to
create the balanced Normal-vs-Pneumonia subset.

Note: ~3.5 GB; the mirror occasionally rate-limits, so the download may take a while and
auto-retries. This dataset is used only for the exploratory external-validation section.

Usage:
    python scripts/fetch_rsna_hf.py
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

REPO = "Baldezo313/rsna-pneumonia-dataset"


def main() -> None:
    print(f"[rsna] downloading {REPO} train DICOMs + label CSVs (~3.5 GB) ...")
    path = snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=["stage_2_train_images_*/*", "*.csv"],
        local_dir="data/raw/rsna",
    )
    print(f"[rsna] done -> {path}")


if __name__ == "__main__":
    main()
