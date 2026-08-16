"""Generate all thesis/paper tables and textual report artifacts.

Every table is exported in three formats — **CSV** (data), **Markdown** (thesis
body), and **LaTeX** (IEEE paper) — from a single source so they can never drift
apart. Tables that depend on experimental results are generated only when the
corresponding ``results/metrics/*.json`` files exist; missing inputs are reported,
not fabricated.

Produces:
    Table 1 Dataset Statistics      Table 4 Final Performance
    Table 2 Training Configuration  Table 5 Quantization Results
    Table 3 Model Comparison        Table 6 Computational Efficiency
    + literature_comparison template, limitations.md, future_work.md, paper_assets/

Run:
    python -m src.reporting --config configs/efficientnet_b0.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, add_config_cli_args, config_from_cli
from .dataset import compute_dataset_statistics
from .utils import ensure_dir, get_logger, load_json

logger = get_logger("reporting")

__all__ = ["save_table", "build_all_reports"]

BASELINE_EXPERIMENTS = ("efficientnet_b0", "resnet18", "mobilenetv3_small")


# --------------------------------------------------------------------------- #
# Table export
# --------------------------------------------------------------------------- #
def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra deps)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_table(
    df: pd.DataFrame, stem: str, out_dir: str | Path, caption: str = "", label: str = ""
) -> dict[str, str]:
    """Save a DataFrame as CSV + Markdown + LaTeX; return the written paths."""
    out_dir = ensure_dir(out_dir)
    paths: dict[str, str] = {}

    csv_path = Path(out_dir) / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)

    md_path = Path(out_dir) / f"{stem}.md"
    md = (f"**{caption}**\n\n" if caption else "") + _df_to_markdown(df)
    md_path.write_text(md, encoding="utf-8")
    paths["md"] = str(md_path)

    tex_path = Path(out_dir) / f"{stem}.tex"
    try:
        tex = df.to_latex(index=False, caption=caption or None, label=label or None, escape=True)
    except Exception:  # noqa: BLE001 - fall back to a minimal tabular
        tex = df.to_latex(index=False)
    tex_path.write_text(tex, encoding="utf-8")
    paths["tex"] = str(tex_path)

    return paths


def _maybe_load(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# Individual tables
# --------------------------------------------------------------------------- #
def table_dataset_statistics(cfg: Config, out_dir: Path) -> dict[str, str]:
    """Table 1: per-split class counts and balance."""
    stats = compute_dataset_statistics(cfg)
    df = pd.DataFrame(stats["per_split"])
    df.columns = ["Split", "Normal", "Pneumonia", "Total", "Pneumonia ratio"]
    return save_table(df, "table1_dataset_statistics", out_dir, "Dataset Statistics", "tab:dataset")


def table_training_config(cfg: Config, out_dir: Path) -> dict[str, str]:
    """Table 2: the key training configuration / hyperparameters."""
    rows = [
        ("Backbone", cfg.model.name),
        ("Pretrained", "ImageNet" if cfg.model.pretrained else "No"),
        ("Image size", f"{cfg.data.img_size}x{cfg.data.img_size}"),
        ("Batch size", cfg.data.batch_size),
        ("Optimizer", "AdamW"),
        ("Learning rate", cfg.train.lr),
        ("Weight decay", cfg.train.weight_decay),
        ("LR schedule", f"{cfg.train.warmup_epochs}-epoch warmup + cosine"),
        ("Epochs", cfg.train.epochs),
        ("Unfreeze epoch", cfg.train.unfreeze_epoch),
        ("Label smoothing", cfg.train.label_smoothing),
        ("Early stopping patience", cfg.train.early_stopping_patience),
        ("Gradient clipping", cfg.train.grad_clip),
        ("Mixed precision", cfg.train.amp),
        ("Seed", cfg.seed),
    ]
    df = pd.DataFrame(rows, columns=["Hyperparameter", "Value"])
    return save_table(df, "table2_training_config", out_dir, "Training Configuration", "tab:trainconfig")


def _param_counts() -> dict[str, int]:
    """Architecture parameter counts (built offline, pretrained=False)."""
    from .models import build_model

    counts: dict[str, int] = {}
    for name in BASELINE_EXPERIMENTS:
        try:
            model = build_model(name=name, pretrained=False, freeze_stages=0)
            counts[name] = sum(p.numel() for p in model.parameters())
        except Exception:  # noqa: BLE001
            counts[name] = 0
    return counts


def table_model_comparison(metrics_dir: Path, out_dir: Path) -> dict[str, str] | None:
    """Table 3: baseline comparison across architectures (requires per-model test metrics)."""
    params = _param_counts()
    rows: list[dict[str, Any]] = []
    for name in BASELINE_EXPERIMENTS:
        test = _maybe_load(metrics_dir / f"{name}_test_metrics.json")
        if test is None:
            continue
        m = test["metrics"]
        bench = _maybe_load(metrics_dir / f"{name}_benchmark.json")
        latency = bench[0]["latency_ms_mean"] if bench else None
        size = bench[0]["size_mb"] if bench else None
        rows.append(
            {
                "Model": name,
                "Params (M)": round(params.get(name, 0) / 1e6, 2),
                "AUC": round(m["auc"], 4),
                "Accuracy": round(m["accuracy"], 4),
                "Sensitivity": round(m["sensitivity"], 4),
                "Specificity": round(m["specificity"], 4),
                "F1": round(m["f1"], 4),
                "Latency (ms)": latency if latency is not None else "—",
                "Size (MB)": size if size is not None else "—",
            }
        )
    if not rows:
        logger.warning("Table 3 skipped: no per-model test_metrics.json found.")
        return None
    return save_table(pd.DataFrame(rows), "table3_model_comparison", out_dir, "Model Comparison", "tab:models")


def table_final_performance(cfg: Config, metrics_dir: Path, out_dir: Path) -> dict[str, str] | None:
    """Table 4: final test performance of the primary model with 95% CIs."""
    test = _maybe_load(metrics_dir / f"{cfg.experiment_name}_test_metrics.json")
    if test is None:
        logger.warning("Table 4 skipped: %s_test_metrics.json not found.", cfg.experiment_name)
        return None
    m, ci = test["metrics"], test["confidence_intervals"]

    def fmt_ci(d: dict[str, float]) -> str:
        if math.isnan(d["low"]):
            return "—"
        return f"[{d['low']:.4f}, {d['high']:.4f}]"

    rows = [
        {"Metric": "ROC-AUC", "Value": round(m["auc"], 4), "95% CI": fmt_ci(ci["auc"])},
        {"Metric": "Accuracy", "Value": round(m["accuracy"], 4), "95% CI": fmt_ci(ci["accuracy"])},
        {"Metric": "Sensitivity (Recall)", "Value": round(m["sensitivity"], 4), "95% CI": "—"},
        {"Metric": "Specificity", "Value": round(m["specificity"], 4), "95% CI": "—"},
        {"Metric": "Precision", "Value": round(m["precision"], 4), "95% CI": "—"},
        {"Metric": "F1-score", "Value": round(m["f1"], 4), "95% CI": "—"},
    ]
    return save_table(pd.DataFrame(rows), "table4_final_performance", out_dir, "Final Test Performance", "tab:final")


def table_quantization(cfg: Config, metrics_dir: Path, out_dir: Path) -> dict[str, str] | None:
    """Table 5: FP32 vs INT8 (dynamic / static) accuracy and size."""
    quant = _maybe_load(metrics_dir / f"{cfg.experiment_name}_quantization.json")
    if quant is None:
        logger.warning("Table 5 skipped: %s_quantization.json not found.", cfg.experiment_name)
        return None
    rows = []
    for v in quant["variants"]:
        rows.append(
            {
                "Variant": v["variant"],
                "Size (MB)": v["size_mb"],
                "Size ↓ (%)": v.get("size_reduction_pct", 0.0),
                "Accuracy": v["accuracy"],
                "Acc drop (%)": v.get("acc_drop_pct", 0.0),
                "AUC": v["auc"],
                "AUC drop (%)": v.get("auc_drop_pct", 0.0),
                "Latency (ms)": v["latency_ms_mean"],
            }
        )
    caption = f"Quantization Results (backend: {quant.get('backend', 'n/a')})"
    return save_table(pd.DataFrame(rows), "table5_quantization", out_dir, caption, "tab:quant")


def table_efficiency(cfg: Config, metrics_dir: Path, out_dir: Path) -> dict[str, str] | None:
    """Table 6: computational efficiency (size, latency, throughput, peak RSS)."""
    bench = _maybe_load(metrics_dir / f"{cfg.experiment_name}_benchmark.json")
    quant = _maybe_load(metrics_dir / f"{cfg.experiment_name}_quantization.json")
    rows: list[dict[str, Any]] = []
    if bench:
        for r in bench:
            rows.append(
                {
                    "Variant": r["variant"],
                    "Size (MB)": r["size_mb"],
                    "Latency mean (ms)": r["latency_ms_mean"],
                    "Latency p95 (ms)": r.get("latency_ms_p95", "—"),
                    "Throughput (img/s)": r.get("throughput_img_per_s", "—"),
                    "Peak RSS Δ (MB)": r.get("peak_rss_delta_mb", "—"),
                }
            )
    elif quant:
        for v in quant["variants"]:
            rows.append(
                {
                    "Variant": v["variant"],
                    "Size (MB)": v["size_mb"],
                    "Latency mean (ms)": v["latency_ms_mean"],
                    "Latency p95 (ms)": "—",
                    "Throughput (img/s)": "—",
                    "Peak RSS Δ (MB)": "—",
                }
            )
    if not rows:
        logger.warning("Table 6 skipped: no benchmark/quantization results found.")
        return None
    return save_table(pd.DataFrame(rows), "table6_efficiency", out_dir, "Computational Efficiency", "tab:efficiency")


def literature_comparison_template(out_dir: Path) -> dict[str, str]:
    """A blank, manually-completed literature-comparison table (research-gap analysis)."""
    columns = [
        "Author", "Year", "Dataset", "Model", "Accuracy", "AUC",
        "Explainability", "Quantization", "Edge Deployment Focus",
    ]
    # One example row pointing at THIS work; the rest are left blank for manual fill.
    example = {
        "Author": "This work", "Year": 2026, "Dataset": "Kermany (pediatric CXR)",
        "Model": "EfficientNet-B0 + Grad-CAM++ + INT8", "Accuracy": "{{see Table 4}}",
        "AUC": "{{see Table 4}}", "Explainability": "Integrated Grad-CAM++",
        "Quantization": "Dynamic + static PTQ", "Edge Deployment Focus": "Yes",
    }
    blanks = [dict.fromkeys(columns, "") for _ in range(6)]
    df = pd.DataFrame([example] + blanks, columns=columns)
    return save_table(
        df, "literature_comparison_template", out_dir,
        "Literature Comparison (template — complete from your reading)", "tab:lit",
    )


# --------------------------------------------------------------------------- #
# Textual artifacts
# --------------------------------------------------------------------------- #
LIMITATIONS_MD = """# Limitations

These limitations are stated explicitly and must not be hidden in the thesis or paper.

1. **Single-dataset evaluation.** All training and testing use the Kermany
   pediatric chest X-ray dataset from a single institution. Results may not
   transfer to other scanners, acquisition protocols, or hospitals.

2. **Limited external validation.** The model was evaluated zero-shot on one
   independent RSNA dataset using a simplified binary label mapping. This is
   supplementary evidence only; it does not replace multi-dataset,
   patient-level, or prospective clinical validation. NIH ChestX-ray14 and
   CheXpert were not evaluated.

3. **No real-world clinical validation.** No prospective study or reader study
   with practising radiologists was conducted. Grad-CAM++ outputs are model-derived
   visualisations, not clinically validated explanations. (No radiologist ratings
   are simulated anywhere in this project.)

4. **No edge-hardware validation.** Efficiency metrics are measured on a
   general-purpose CPU as a proxy for edge devices. Latency, memory and power on
   actual Raspberry Pi / Jetson hardware were not measured.

5. **Potential dataset bias.** The Kermany dataset is pediatric and class-imbalanced
   (pneumonia-heavy). Performance across age, sex, ethnicity and comorbidity
   subgroups was not assessed and may be biased.

Additional methodological notes:
- The official Kermany validation split (16 images) is too small for model
  selection; a stratified split carved from the training set is used instead, with
  the official test set held out untouched.
- Static INT8 quantization is backend-dependent; if unsupported on the host, the
  pipeline reports this rather than substituting estimated numbers.
"""

FUTURE_WORK_MD = """# Future Work

These directions extend naturally into a PhD research programme on trustworthy,
resource-efficient medical AI.

1. **External validation on NIH ChestX-ray14.** Assess generalisation to a large,
   adult, multi-pathology dataset and quantify domain shift.

2. **Cross-dataset validation on CheXpert.** Evaluate against expert-labelled,
   uncertainty-aware annotations and study calibration under distribution shift.

3. **Edge-hardware deployment.** Benchmark the quantized model on Raspberry Pi 4
   and NVIDIA Jetson Nano for real latency, memory, energy and thermal behaviour.

4. **TinyML optimisation.** Explore quantization-aware training, structured
   pruning, and knowledge distillation to push below microcontroller-class budgets.

5. **Federated learning.** Train across institutions without sharing raw images to
   improve generalisation while preserving patient privacy.

6. **Multi-disease chest X-ray diagnosis.** Extend from binary pneumonia detection
   to multi-label thoracic disease classification with calibrated, explainable
   per-finding outputs.
"""


def write_text_artifacts(reports_dir: Path) -> dict[str, str]:
    """Write limitations.md and future_work.md."""
    reports_dir = ensure_dir(reports_dir)
    lim = reports_dir / "limitations.md"
    fut = reports_dir / "future_work.md"
    lim.write_text(LIMITATIONS_MD, encoding="utf-8")
    fut.write_text(FUTURE_WORK_MD, encoding="utf-8")
    return {"limitations": str(lim), "future_work": str(fut)}


def write_paper_assets(cfg: Config, paper_dir: Path, metrics_dir: Path) -> dict[str, str]:
    """Emit figure/table captions and a result-summary stub for the IEEE paper."""
    paper_dir = ensure_dir(paper_dir)
    captions = """# Figure & Table Captions (IEEE paper)

## Figures
- Fig. 1. Class distribution across train/validation/test splits of the Kermany dataset.
- Fig. 2. Training and validation loss / ROC-AUC curves and learning-rate schedule.
- Fig. 3. Confusion matrix on the held-out test set.
- Fig. 4. ROC curve (with AUC) on the held-out test set.
- Fig. 5. Precision-Recall curve on the held-out test set.
- Fig. 6. Grad-CAM++ overlays for correct, false-positive and false-negative cases.
- Fig. 7. FP32 vs INT8 (dynamic / static) comparison: size, latency, AUC.
- Fig. 8. Model size comparison across architectures and quantization variants.

## Tables
- Table I. Dataset statistics (see results/tables/table1_dataset_statistics.tex).
- Table II. Training configuration (table2_training_config.tex).
- Table III. Model comparison across architectures (table3_model_comparison.tex).
- Table IV. Final test performance with 95% bootstrap CIs (table4_final_performance.tex).
- Table V. Quantization results (table5_quantization.tex).
- Table VI. Computational efficiency (table6_efficiency.tex).
"""
    (paper_dir / "captions.md").write_text(captions, encoding="utf-8")

    test = _maybe_load(metrics_dir / f"{cfg.experiment_name}_test_metrics.json")
    if test:
        m = test["metrics"]
        summary = (
            f"# Result Summary ({cfg.experiment_name})\n\n"
            f"- Test AUC: {m['auc']:.4f}\n"
            f"- Accuracy: {m['accuracy']:.4f}\n"
            f"- Sensitivity: {m['sensitivity']:.4f}\n"
            f"- Specificity: {m['specificity']:.4f}\n"
            f"- F1: {m['f1']:.4f}\n"
        )
    else:
        summary = (
            "# Result Summary\n\n"
            "_Results pending — run `make reproduce` (train -> evaluate -> quantize -> "
            "benchmark -> explain -> report) on the Kermany dataset to populate._\n"
        )
    (paper_dir / "results_summary.md").write_text(summary, encoding="utf-8")
    return {"captions": str(paper_dir / "captions.md"), "summary": str(paper_dir / "results_summary.md")}


def build_all_reports(cfg: Config) -> dict[str, Any]:
    """Generate every table and report artifact that the available results allow."""
    results_dir = Path(cfg.paths.results_dir)
    tables_dir = ensure_dir(results_dir / "tables")
    metrics_dir = results_dir / "metrics"
    reports_dir = Path("reports")
    paper_dir = Path("paper_assets")

    generators = {
        "table1": lambda: table_dataset_statistics(cfg, tables_dir),
        "table2": lambda: table_training_config(cfg, tables_dir),
        "table3": lambda: table_model_comparison(metrics_dir, tables_dir),
        "table4": lambda: table_final_performance(cfg, metrics_dir, tables_dir),
        "table5": lambda: table_quantization(cfg, metrics_dir, tables_dir),
        "table6": lambda: table_efficiency(cfg, metrics_dir, tables_dir),
        "literature_template": lambda: literature_comparison_template(tables_dir),
        "text_artifacts": lambda: write_text_artifacts(reports_dir),
        "paper_assets": lambda: write_paper_assets(cfg, paper_dir, metrics_dir),
    }
    written: dict[str, Any] = {}
    for key, gen in generators.items():
        try:
            written[key] = gen()
        except Exception as exc:  # noqa: BLE001 - one missing input must not abort the rest
            logger.warning("Skipped %s: %s", key, exc)
            written[key] = None
    logger.info("Reports generated under %s, reports/ and paper_assets/", tables_dir)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis/paper tables and report artifacts.")
    add_config_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_cli(args)
    build_all_reports(cfg)


if __name__ == "__main__":
    main()
