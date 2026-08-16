"""Verify that the headline numbers in reports/thesis_final.md are traceable to results.

Loads the results JSON, formats stable headline numbers exactly as the thesis reports
them, and checks each string is present in the thesis. Process-RSS and timing values are
reported separately because they vary with the host and container runtime; they are not
used as exact text checks. Exits non-zero if any stable check fails.

Usage:
    python scripts/verify_thesis_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "reports" / "thesis_final.md"
METRICS = ROOT / "results" / "metrics"


def _load(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def main() -> int:
    if not THESIS.exists():
        print(f"[verify] thesis not found: {THESIS}")
        return 2
    # Normalise: drop thousands separators so '3,140.9' matches '3140.9'.
    text = THESIS.read_text(encoding="utf-8").replace(",", "")

    tm = _load("efficientnet_b0_test_metrics.json")
    m, ci = tm["metrics"], tm["confidence_intervals"]
    quant = {v["variant"]: v for v in _load("efficientnet_b0_quantization.json")["variants"]}
    mp = _load("memory_profile.json")
    sv = mp["streaming_vs_naive"]
    rt = {v["variant"]: v for v in mp["runtime_memory"]["variants"]}
    dv = _load("dataset_validation.json")
    stat = quant["INT8 static (PTQ)"]

    checks: list[tuple[str, str]] = [
        ("test ROC-AUC", f"{m['auc']:.4f}"),
        ("AUC CI low", f"{ci['auc']['low']:.4f}"),
        ("AUC CI high", f"{ci['auc']['high']:.4f}"),
        ("accuracy", f"{m['accuracy']:.4f}"),
        ("sensitivity", f"{m['sensitivity']:.4f}"),
        ("specificity", f"{m['specificity']:.4f}"),
        ("F1", f"{m['f1']:.4f}"),
        ("static INT8 size", f"{stat['size_mb']:.2f}"),
        ("static INT8 AUC drop", f"{stat['auc_drop_pct']:.2f}"),
        ("duplicates removed", f"{dv['n_train_duplicates_removed']} byte-identical"),
    ]

    # These are valid measurements, but exact values can move between runs because
    # allocators, worker scheduling, and the Docker host affect process RSS. Keep them
    # visible in verification output without making the full pipeline fail on harmless
    # one-decimal drift.
    volatile_checks: list[tuple[str, str]] = [
        ("streaming RSS (run-specific)", f"{sv['streaming_peak_rss_mb']:.1f} MB"),
        ("naive RSS (run-specific)", f"{sv['naive_full_ram_peak_rss_mb']:.1f} MB"),
        ("streaming reduction (run-specific)", f"{sv['reduction_factor']:.1f}x"),
        ("FP32 inference RSS (run-specific)", f"{rt['FP32']['inference_peak_rss_mb']:.1f} MB"),
        ("static INT8 inference RSS (run-specific)", f"{rt['INT8 static (PTQ)']['inference_peak_rss_mb']:.1f} MB"),
    ]

    # External validation (RSNA) — included only if that experiment has been run.
    rsna_path = METRICS / "rsna_external_metrics.json"
    if rsna_path.exists():
        r = json.loads(rsna_path.read_text(encoding="utf-8"))
        rm, rci = r["metrics"], r["confidence_intervals"]
        checks += [
            ("RSNA AUC", f"{rm['auc']:.4f}"),
            ("RSNA AUC CI low", f"{rci['auc']['low']:.4f}"),
            ("RSNA AUC CI high", f"{rci['auc']['high']:.4f}"),
            ("RSNA sensitivity", f"{rm['sensitivity']:.4f}"),
            ("RSNA specificity", f"{rm['specificity']:.4f}"),
            ("RSNA F1", f"{rm['f1']:.4f}"),
        ]

    # Statistical analysis (calibration + extended metrics) — only if that step has run.
    stats_path = METRICS / "statistical_analysis.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        for name in ("efficientnet_b0", "resnet18", "mobilenetv3_small"):
            pm = s["per_model_kermany"][name]
            checks += [
                (f"{name} ECE", f"{pm['ece']:.4f}"),
                (f"{name} Brier", f"{pm['brier']:.4f}"),
            ]
        cis = s["primary_full_cis"]
        checks += [
            ("primary MCC", f"{cis['mcc']['point']:.4f}"),
            ("primary balanced accuracy", f"{cis['balanced_accuracy']['point']:.4f}"),
            ("primary sensitivity CI low", f"{cis['sensitivity']['low']:.4f}"),
            ("primary specificity CI low", f"{cis['specificity']['low']:.4f}"),
        ]

    failures = []
    for label, expected in checks:
        ok = expected in text
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: '{expected}'")
        if not ok:
            failures.append((label, expected))

    for label, measured in volatile_checks:
        print(f"  [MEASURED] {label}: {measured}; compare with the run record, not exact thesis text")

    print()
    if failures:
        print(f"[verify] {len(failures)} number(s) NOT found verbatim in the thesis — "
              f"update reports/thesis_final.md or regenerate results.")
        return 1
    print(f"[verify] all {len(checks)} stable headline numbers in the thesis match results/metrics/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
