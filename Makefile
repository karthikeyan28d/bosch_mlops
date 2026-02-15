# Makefile for Biometric MLOps Project
# Provides convenient commands for development workflow

.PHONY: install install-dev test lint format clean train preprocess inference databricks-validate databricks-deploy

# Python settings
PYTHON := python
PIP := pip

# Default target
all: install test

# ============================================================================
# Installation
# ============================================================================

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,ray,mlflow]"

install-all:
	$(PIP) install -e ".[all]"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# ============================================================================
# Testing
# ============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# ============================================================================
# ML Pipeline
# ============================================================================

preprocess:
	$(PYTHON) -m src.preprocessing --config configs/config.yaml

train:
	$(PYTHON) -m src.train --config configs/config.yaml

inference:
	$(PYTHON) -m src.inference --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt

benchmark:
	$(PYTHON) -m src.preprocessing --benchmark --data-dir data/raw

# ============================================================================
# Databricks Asset Bundles
# ============================================================================

databricks-validate:
	databricks bundle validate -t dev

databricks-deploy-dev:
	databricks bundle deploy -t dev

databricks-deploy-prod:
	databricks bundle deploy -t prod

databricks-run-training:
	databricks bundle run biometric_training -t dev

# ============================================================================
# Build & Clean
# ============================================================================

build:
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-outputs:
	rm -rf outputs/ mlruns/

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "Architecture and design docs are in README.md and docs/"

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Available commands:"
	@echo "  make install          - Install package"
	@echo "  make install-dev      - Install with dev dependencies"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make lint             - Run linting"
	@echo "  make format           - Format code"
	@echo "  make train            - Run training pipeline"
	@echo "  make preprocess       - Run data preprocessing"
	@echo "  make inference        - Run inference pipeline"
	@echo "  make benchmark        - Benchmark preprocessing backends"
	@echo "  make databricks-validate  - Validate Databricks bundle"
	@echo "  make databricks-deploy-dev - Deploy to dev workspace"
	@echo "  make clean            - Clean build artifacts"
