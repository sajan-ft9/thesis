"""Run the primary EfficientNet-B0 robustness experiment for multiple seeds.

Each seed gets its own checkpoint, test metrics, quantization artifacts, and metadata.
The held-out Kermany test set and the 0.5 threshold remain unchanged; the test set is
never used for training, calibration, or model selection.

Example:
    python scripts/run_seed_sensitivity.py --config configs/efficientnet_b0.yaml \
        --seeds 42 43 44 --device cpu
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from src.config import load_config
from src.evaluate import evaluate_checkpoint
from src.quantize import run_quantization_study
from src.train import train
from src.utils import save_json


def run_seed_sensitivity(config_path: str, seeds: list[int], device: str) -> dict[str, Any]:
    """Train, evaluate, and quantize the primary model once for each seed."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        experiment_name = f"efficientnet_b0_seed{seed}"
        cfg = load_config(
            config_path,
            overrides=[f"seed={seed}", f"experiment_name={experiment_name}", f"device={device}"],
        )
        training = train(cfg)
        checkpoint = Path(training["checkpoint"])
        evaluation = evaluate_checkpoint(checkpoint, cfg=cfg, device_str=device)
        quantization = run_quantization_study(checkpoint, cfg=cfg)
        rows.append(
            {
                "seed": seed,
                "experiment_name": experiment_name,
                "checkpoint": str(checkpoint),
                "test_metrics": evaluation["metrics"],
                "test_confidence_intervals": evaluation["confidence_intervals"],
                "threshold": evaluation["threshold"],
                "threshold_source": evaluation["threshold_source"],
                "quantization": quantization,
            }
        )
        # Release the previous model before starting the next seed in the same
        # process. This matters on CPU-only machines with limited RAM.
        gc.collect()

    output = {
        "experiment": "efficientnet_b0_seed_sensitivity",
        "seeds": seeds,
        "device": device,
        "test_set_policy": "same held-out Kermany test set; no test-time tuning",
        "rows": rows,
    }
    out_path = Path("results/metrics/efficientnet_b0_seed_sensitivity.json")
    save_json(output, out_path)
    print(f"[seed-sensitivity] saved {out_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_seed_sensitivity(args.config, args.seeds, args.device)


if __name__ == "__main__":
    main()
