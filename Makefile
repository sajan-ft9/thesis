# Makefile — reproducible entry points for the pneumonia-edge-xai project.
# Override the interpreter with:  make <target> PYTHON=python
PYTHON ?= .venv/bin/python
PRIMARY ?= configs/efficientnet_b0.yaml
CKPT ?= models/efficientnet_b0_best.pth

.DEFAULT_GOAL := help
.PHONY: help setup test lint smoke validate-data train train-baselines \
        evaluate benchmark quantize explain report render reproduce clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:  ## Run the unit + integration test suite
	$(PYTHON) -m pytest tests/

lint:  ## Lint with ruff (if installed)
	$(PYTHON) -m ruff check src tests scripts || true

smoke:  ## End-to-end pipeline on tiny SYNTHETIC data (no real numbers)
	bash scripts/run_smoke_test.sh

validate-data:  ## Validate the real dataset (class balance, duplicates, leakage)
	$(PYTHON) -m src.dataset --config $(PRIMARY)

train:  ## Train the primary model (EfficientNet-B0)
	$(PYTHON) -m src.train --config $(PRIMARY)

train-baselines:  ## Train ResNet-18 and MobileNetV3-Small baselines
	$(PYTHON) -m src.train --config configs/resnet18.yaml
	$(PYTHON) -m src.train --config configs/mobilenetv3.yaml

evaluate:  ## Evaluate all trained checkpoints on the test set
	$(PYTHON) -m src.evaluate --checkpoint models/efficientnet_b0_best.pth
	$(PYTHON) -m src.evaluate --checkpoint models/resnet18_best.pth
	$(PYTHON) -m src.evaluate --checkpoint models/mobilenetv3_small_best.pth

benchmark:  ## Benchmark efficiency (size, latency, throughput, RSS) for all models
	$(PYTHON) -m src.benchmarking --checkpoint models/efficientnet_b0_best.pth
	$(PYTHON) -m src.benchmarking --checkpoint models/resnet18_best.pth
	$(PYTHON) -m src.benchmarking --checkpoint models/mobilenetv3_small_best.pth

quantize:  ## Quantization study (dynamic + static PTQ) on the primary model
	$(PYTHON) -m src.quantize --checkpoint $(CKPT)

explain:  ## Generate Grad-CAM++ explanations for correct / FP / FN cases
	$(PYTHON) -m src.explainability --checkpoint $(CKPT)

memory:  ## Memory-footprint analysis (streaming vs naive RAM; per-precision RSS)
	$(PYTHON) -m src.memory_profile --config $(PRIMARY) --checkpoint $(CKPT)

report:  ## Generate all tables + report artifacts from results
	$(PYTHON) -m src.reporting --config $(PRIMARY)

render:  ## Fill thesis placeholders with computed results
	$(PYTHON) scripts/render_report.py --config $(PRIMARY)

stats:  ## Deeper statistics: paired-bootstrap AUC comparison + calibration (ECE/Brier), inference-only
	$(PYTHON) -m src.statistical_analysis

tradeoff:  ## Plot the accuracy-efficiency trade-off (deployment frontier) figure
	$(PYTHON) scripts/plot_tradeoff.py

verify-numbers:  ## Check thesis_final.md headline numbers match results/metrics/*.json
	$(PYTHON) scripts/verify_thesis_numbers.py

kfold:  ## 5-fold CV on the primary model, each fold evaluated on the held-out test set (~2-3h; not part of `reproduce`)
	$(PYTHON) scripts/run_kfold_cv.py --config $(PRIMARY) --folds 5

external-rsna:  ## Exploratory zero-shot external validation on RSNA (download + inference only)
	$(PYTHON) scripts/fetch_rsna_hf.py
	$(PYTHON) scripts/build_rsna_subset.py --seed 42
	$(PYTHON) -m src.evaluate --checkpoint $(CKPT) --external-dir data/processed/rsna_external --tag rsna_external --device cpu

reproduce: validate-data train train-baselines evaluate benchmark quantize explain memory stats tradeoff report render  ## Full pipeline on the real dataset

clean:  ## Remove generated results (keeps directory structure)
	find results -type f ! -name '.gitkeep' -delete
	find models -type f ! -name '.gitkeep' -delete
	rm -f reports/thesis_rendered.md
