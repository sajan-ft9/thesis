# Makefile — reproducible entry points for the pneumonia-edge-xai project.
# Override the interpreter with:  make <target> PYTHON=python
PYTHON ?= .venv/bin/python
PRIMARY ?= configs/efficientnet_b0.yaml
CKPT ?= models/efficientnet_b0_best.pth
COMPOSE ?= docker compose
DOCKER_SERVICE ?= real-validate
DOCKER_RUN = $(COMPOSE) run --rm $(DOCKER_SERVICE)

.DEFAULT_GOAL := help
.PHONY: help setup test lint smoke validate-data train train-baselines \
        evaluate benchmark quantize explain memory stats tradeoff report render \
        verify-numbers external-rsna reproduce clean \
        docker-build docker-test docker-smoke docker-lint docker-validate-data \
        docker-train docker-train-baselines docker-evaluate docker-benchmark \
        docker-quantize docker-explain docker-memory docker-stats docker-tradeoff \
        docker-report docker-render docker-verify docker-external-rsna \
        docker-inference docker-actual docker-reproduce docker-publication

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

external-rsna:  ## Exploratory zero-shot external validation on RSNA (download + inference only)
	$(PYTHON) scripts/fetch_rsna_hf.py
	$(PYTHON) scripts/build_rsna_subset.py --seed 42
	$(PYTHON) -m src.evaluate --checkpoint $(CKPT) --external-dir data/processed/rsna_external --tag rsna_external --device cpu

reproduce: validate-data train train-baselines evaluate benchmark quantize explain memory stats tradeoff report render  ## Full pipeline on the real dataset

# Docker targets are the canonical portable entry points. `docker-test` and
# `docker-smoke` are software tests and use synthetic fixtures; they must never
# be used as research results. Targets prefixed with `docker-actual` operate on
# the mounted real datasets and existing checkpoints.

docker-build:  ## Build the CPU Docker image
	$(COMPOSE) build

docker-test:  ## Run unit/integration tests; synthetic fixtures only, no research numbers
	$(COMPOSE) run --rm test

docker-smoke:  ## Run the tiny end-to-end smoke test; synthetic fixtures only
	$(COMPOSE) run --rm smoke

docker-lint:  ## Run Ruff inside Docker
	$(DOCKER_RUN) python -m ruff check src tests scripts

docker-validate-data:  ## Validate the mounted real Kermany dataset and write dataset_validation.json
	$(DOCKER_RUN) python -m src.dataset --config $(PRIMARY)

docker-train:  ## Train the primary EfficientNet-B0 on the real Kermany dataset
	$(DOCKER_RUN) python -m src.train --config $(PRIMARY)

docker-train-baselines:  ## Train real-data ResNet-18 and MobileNetV3-Small baselines
	$(DOCKER_RUN) python -m src.train --config configs/resnet18.yaml
	$(DOCKER_RUN) python -m src.train --config configs/mobilenetv3.yaml

docker-evaluate:  ## Evaluate existing real-data checkpoints on the locked Kermany test set
	$(DOCKER_RUN) python -m src.evaluate --checkpoint /app/models/efficientnet_b0_best.pth --device cpu
	$(DOCKER_RUN) python -m src.evaluate --checkpoint /app/models/resnet18_best.pth --device cpu
	$(DOCKER_RUN) python -m src.evaluate --checkpoint /app/models/mobilenetv3_small_best.pth --device cpu

docker-benchmark:  ## Measure real checkpoint size, CPU latency, throughput, and RSS
	$(DOCKER_RUN) python -m src.benchmarking --checkpoint /app/models/efficientnet_b0_best.pth
	$(DOCKER_RUN) python -m src.benchmarking --checkpoint /app/models/resnet18_best.pth
	$(DOCKER_RUN) python -m src.benchmarking --checkpoint /app/models/mobilenetv3_small_best.pth

docker-quantize:  ## Quantize the real EfficientNet checkpoint and evaluate FP32/dynamic/static variants
	$(DOCKER_RUN) python -m src.quantize --checkpoint /app/models/efficientnet_b0_best.pth

docker-explain:  ## Generate qualitative Grad-CAM++ outputs from real test images
	$(DOCKER_RUN) python -m src.explainability --checkpoint /app/models/efficientnet_b0_best.pth

docker-memory:  ## Measure real streaming/naive loader RSS and isolated per-variant inference RSS
	$(DOCKER_RUN) python -m src.memory_profile --config $(PRIMARY) --checkpoint /app/models/efficientnet_b0_best.pth

docker-stats:  ## Compute real held-out paired comparisons, calibration, and bootstrap statistics
	$(DOCKER_RUN) python -m src.statistical_analysis

docker-tradeoff:  ## Plot the real accuracy-efficiency deployment frontier
	$(DOCKER_RUN) python scripts/plot_tradeoff.py

docker-report:  ## Generate tables, figures, and report artifacts from current real metrics
	$(DOCKER_RUN) python -m src.reporting --config $(PRIMARY)

docker-render:  ## Render the thesis copy from reports/thesis_final.md and current real metrics
	$(DOCKER_RUN) python scripts/render_report.py --config $(PRIMARY) --thesis reports/thesis_final.md --out reports/thesis_rendered.md

docker-verify:  ## Verify thesis headline numbers against current real result JSON files
	$(DOCKER_RUN) python scripts/verify_thesis_numbers.py

docker-external-rsna:  ## Fetch/build the real RSNA probe and run zero-shot external evaluation
	$(DOCKER_RUN) python scripts/fetch_rsna_hf.py
	$(DOCKER_RUN) python scripts/build_rsna_subset.py --seed 42
	$(DOCKER_RUN) python -m src.evaluate --checkpoint /app/models/efficientnet_b0_best.pth --external-dir /app/data/processed/rsna_external --tag rsna_external --device cpu

docker-inference:  ## Run one-image inference for input/image.png using the real checkpoint
	$(COMPOSE) run --rm inference

docker-actual: docker-validate-data docker-evaluate docker-benchmark docker-quantize docker-explain docker-memory docker-stats docker-tradeoff docker-report docker-render docker-verify  ## Run the actual real-data analysis using existing checkpoints

docker-reproduce: docker-validate-data docker-train docker-train-baselines docker-evaluate docker-benchmark docker-quantize docker-explain docker-memory docker-stats docker-tradeoff docker-report docker-render docker-verify  ## Train and run the complete actual real-data pipeline

docker-publication: docker-actual docker-external-rsna docker-report docker-render docker-verify  ## Rebuild the publication package including the optional RSNA probe

clean:  ## Remove generated results (keeps directory structure)
	find results -type f ! -name '.gitkeep' -delete
	find models -type f ! -name '.gitkeep' -delete
	rm -f reports/thesis_rendered.md
