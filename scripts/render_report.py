"""Substitute ``{{token}}`` placeholders in reports/thesis.md with real results.

The committed thesis (``reports/thesis.md``) contains NO hard-coded numbers; it
uses ``{{token}}`` placeholders. After a real run produces ``results/metrics/*.json``,
this script fills the placeholders and writes ``reports/thesis_rendered.md``.

Any placeholder without a corresponding result is left intact and counted as
"pending" — so an un-run thesis is obviously incomplete rather than silently wrong.

Usage:
    python scripts/render_report.py --config configs/efficientnet_b0.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.dataset import compute_dataset_statistics  # noqa: E402


def _load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _fmt(value: float, nd: int = 4) -> str:
    return f"{value:.{nd}f}"


def build_values(config_path: str) -> dict[str, str]:
    """Assemble the token -> value mapping from available result files."""
    cfg = load_config(config_path)
    metrics_dir = Path(cfg.paths.results_dir) / "metrics"
    name = cfg.experiment_name
    values: dict[str, str] = {}

    # Dataset statistics (always computable if data present).
    try:
        stats = compute_dataset_statistics(cfg)
        per = {r["split"]: r for r in stats["per_split"]}
        values.update(
            ds_train=str(per["train"]["total"]),
            ds_val=str(per["val"]["total"]),
            ds_test=str(per["test"]["total"]),
            ds_total=str(per["total"]["total"]),
            ds_normal=str(per["total"]["normal"]),
            ds_pneumonia=str(per["total"]["pneumonia"]),
        )
    except Exception:  # noqa: BLE001 - dataset may be absent at render time
        pass

    test = _load(metrics_dir / f"{name}_test_metrics.json")
    if test:
        m, ci = test["metrics"], test["confidence_intervals"]
        values.update(
            test_auc=_fmt(m["auc"]),
            test_acc=_fmt(m["accuracy"]),
            test_sens=_fmt(m["sensitivity"]),
            test_spec=_fmt(m["specificity"]),
            test_precision=_fmt(m["precision"]),
            test_f1=_fmt(m["f1"]),
            n_test=str(test["n_test"]),
            n_errors=str(test["n_errors"]),
            n_fp=str(test["n_false_positive"]),
            n_fn=str(test["n_false_negative"]),
        )
        if ci["auc"]["low"] == ci["auc"]["low"]:
            values["test_auc_ci"] = f"[{_fmt(ci['auc']['low'])}, {_fmt(ci['auc']['high'])}]"
        if ci["accuracy"]["low"] == ci["accuracy"]["low"]:
            values["test_acc_ci"] = f"[{_fmt(ci['accuracy']['low'])}, {_fmt(ci['accuracy']['high'])}]"

    quant = _load(metrics_dir / f"{name}_quantization.json")
    if quant:
        values["quant_backend"] = str(quant.get("backend", "n/a"))
        by_variant = {v["variant"]: v for v in quant["variants"]}
        if "FP32" in by_variant:
            values["fp32_size"] = _fmt(by_variant["FP32"]["size_mb"], 2)
        if "INT8 dynamic" in by_variant:
            values["int8_dyn_size"] = _fmt(by_variant["INT8 dynamic"]["size_mb"], 2)
        if "INT8 static (PTQ)" in by_variant:
            s = by_variant["INT8 static (PTQ)"]
            values["int8_static_size"] = _fmt(s["size_mb"], 2)
            values["int8_static_reduction"] = _fmt(s.get("size_reduction_pct", 0.0), 1)
            values["int8_static_acc_drop"] = _fmt(s.get("acc_drop_pct", 0.0), 3)
            values["int8_static_auc_drop"] = _fmt(s.get("auc_drop_pct", 0.0), 3)

    bench = _load(metrics_dir / f"{name}_benchmark.json")
    if bench:
        values["fp32_latency"] = _fmt(bench[0]["latency_ms_mean"], 2)
        values["fp32_throughput"] = _fmt(bench[0]["throughput_img_per_s"], 1)

    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--thesis", default="reports/thesis.md")
    parser.add_argument("--out", default="reports/thesis_rendered.md")
    args = parser.parse_args()

    values = build_values(args.config)
    text = Path(args.thesis).read_text(encoding="utf-8")

    tokens = set(re.findall(r"\{\{(\w+)\}\}", text))
    filled, pending = 0, []
    for token in tokens:
        if token in values:
            text = text.replace(f"{{{{{token}}}}}", values[token])
            filled += 1
        else:
            pending.append(token)

    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[render] filled {filled} placeholder(s); {len(pending)} pending -> {args.out}")
    if pending:
        print(f"[render] pending tokens (run the pipeline to populate): {sorted(pending)}")


if __name__ == "__main__":
    main()
