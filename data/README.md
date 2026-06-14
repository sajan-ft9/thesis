# Data

This project uses the **Kermany Chest X-Ray Pneumonia dataset** (pediatric CXR,
~5,856 images, classes: `NORMAL`, `PNEUMONIA`). The dataset is **not committed**
to this repository (see licensing below); download it locally before running the
real pipeline.

## Expected layout

After download, the data directory must look like this (default root:
`data/raw/chest_xray`, configurable via `data.root` in `configs/base.yaml`):

```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/*.jpeg
│   └── PNEUMONIA/*.jpeg
└── test/
    ├── NORMAL/*.jpeg
    └── PNEUMONIA/*.jpeg
```

> The original dataset also ships a tiny `val/` split (16 images). This project
> **does not** use it for model selection; instead it carves a stratified
> validation split from `train/` (see `data.val_split`) and keeps the official
> `test/` set untouched for final evaluation.

## Option A — kagglehub (recommended)

```bash
pip install kagglehub
python - <<'PY'
import kagglehub, shutil, pathlib
src = pathlib.Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
# The archive nests a 'chest_xray' folder; locate and link it to data/raw/chest_xray
cand = next(src.rglob("train"), None).parent
dst = pathlib.Path("data/raw/chest_xray"); dst.parent.mkdir(parents=True, exist_ok=True)
print("Found dataset at:", cand)
print("Point configs `data.root` here, or copy it to", dst)
PY
```

## Option B — Kaggle CLI

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw
unzip -q data/raw/chest-xray-pneumonia.zip -d data/raw
# Ensure the final path is data/raw/chest_xray/{train,test}/...
```

## Validate the download

```bash
make validate-data    # checks class balance, byte-duplicate images, and split leakage
```

This writes `results/metrics/dataset_validation.json` and reports whether the
splits are clean.

## Licensing

The Kermany dataset is released under **CC BY 4.0**. Cite the original work:

> Kermany DS, et al. *Identifying Medical Diagnoses and Treatable Diseases by
> Image-Based Deep Learning.* Cell. 2018;172(5):1122-1131.e9.
> doi:10.1016/j.cell.2018.02.010

## Synthetic smoke-test data

`scripts/make_synthetic_data.py` generates a tiny **synthetic** dataset under
`data/synthetic/` for pipeline testing only. These are random images, **not**
chest X-rays — never report any numbers derived from them.
