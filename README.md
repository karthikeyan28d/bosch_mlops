# Multimodal Biometric MLOps Pipeline

A production-quality machine learning infrastructure for multimodal biometric recognition (Iris + Fingerprint) with focus on scalability, reproducibility, and MLOps best practices.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐                   │
│  │   Raw Data   │───▶│  Preprocessing   │───▶│  Feature Store   │                   │
│  │  (Images)    │    │  (Ray Parallel)  │    │  (PyArrow/Delta) │                   │
│  └──────────────┘    └──────────────────┘    └──────────────────┘                   │
│         │                    │                        │                              │
│         │                    ▼                        ▼                              │
│         │           ┌──────────────────┐    ┌──────────────────┐                    │
│         │           │   Data Loader    │◀───│  Config Manager  │                    │
│         │           │   (PyTorch)      │    │    (YAML/Hydra)  │                    │
│         │           └──────────────────┘    └──────────────────┘                    │
│         │                    │                        │                              │
│         │                    ▼                        │                              │
│         │           ┌──────────────────┐              │                              │
│         └──────────▶│  Training Loop   │◀─────────────┘                              │
│                     │  (PyTorch/MLflow)│                                             │
│                     └──────────────────┘                                             │
│                            │                                                         │
│                 ┌──────────┼──────────┐                                              │
│                 ▼          ▼          ▼                                              │
│         ┌────────────┐ ┌────────┐ ┌────────────┐                                    │
│         │   Model    │ │Metrics │ │ Artifacts  │                                    │
│         │  Registry  │ │Logging │ │  Storage   │                                    │
│         └────────────┘ └────────┘ └────────────┘                                    │
│                 │                                                                    │
│                 ▼                                                                    │
│         ┌──────────────────┐    ┌──────────────────┐                                │
│         │    Inference     │───▶│   Predictions    │                                │
│         │    Pipeline      │    │   (Delta/CSV)    │                                │
│         └──────────────────┘    └──────────────────┘                                │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

Deployment Targets:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  LOCAL (VS Code)          │  DATABRICKS (Cloud)                                     │
│  ├── Python venv          │  ├── Databricks Asset Bundles                          │
│  ├── Ray local cluster    │  ├── Unity Catalog                                     │
│  └── MLflow local         │  └── MLflow managed                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. **Modular Architecture**
- Single responsibility principle: each module handles one concern
- Config-driven: all parameters externalized to YAML
- Environment-agnostic: same code runs locally and on Databricks

### 2. **Data Pipeline (Multimodal Handling)**
- **PyTorch Dataset**: Custom dataset class handling iris + fingerprint images
- **PyArrow**: Efficient columnar storage for metadata
- **Ray**: Parallel preprocessing across CPU cores
- **Lazy Loading**: Memory-efficient image loading

### 3. **Reproducibility**
- Seed management for all random operations
- Config versioning with git
- MLflow experiment tracking
- Checkpoint saving/loading

### 4. **Scalability Trade-offs**
| Component | Local | Cloud | Trade-off |
|-----------|-------|-------|-----------|
| Preprocessing | Ray (local cores) | Spark UDFs | Memory vs Speed |
| Training | Single GPU | Distributed | Complexity vs Scale |
| Storage | Local filesystem | Delta Lake | Cost vs Durability |

## Project Structure

```
bosch_mlops/
├── azure-pipelines.yml        # Azure DevOps CI/CD pipeline
├── databricks.yml              # Databricks Asset Bundle config
├── project_config.yml          # Project variables
├── configs/                    # Configuration files
│   └── config.yaml             # Main config
├── configuration/              # Azure pipeline variables
│   └── pipeline_vars/
│       └── vars-config.yml
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_download.py        # Kaggle data download
│   ├── data_loader.py          # PyTorch Dataset & DataLoader
│   ├── preprocessing.py        # Ray-based parallel preprocessing
│   ├── model.py                # PyTorch model architecture
│   ├── model_registry.py       # Unity Catalog model registry
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Model evaluation
│   ├── retrain.py              # Champion/Challenger retraining
│   ├── monitoring.py           # Drift monitoring
│   ├── inference.py            # Inference pipeline
│   └── utils.py                # Utilities
├── resources/                  # Databricks workflows
│   ├── training.yml            # Training job
│   ├── inference.yml           # Inference job
│   └── test_job.yml            # CI test job
├── tests/                      # Unit & integration tests
│   ├── unit/
│   └── integration/
├── docs/                       # Documentation
│   └── DESIGN.md
├── data/                       # Data directory (gitignored)
├── requirements.txt
├── pyproject.toml
├── setup.py
├── run.py
└── README.md
```

## Quick Start

### Local Development

```bash
# 1. Clone and setup
git clone <repo>
cd bosch_mlops
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .

# 2. Setup Kaggle credentials (choose one method)
# Option A: Create kaggle.json file
#   - Go to https://www.kaggle.com/settings
#   - Click 'Create New Token'
#   - Save kaggle.json to ~/.kaggle/kaggle.json

# Option B: Set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# 3. Download dataset automatically
python run.py --mode download

# OR run complete pipeline in one command:
python run.py --mode all

# 4. Individual steps (alternative)
# Preprocess data
python run.py --mode preprocess

# Train model
python run.py --mode train

# Run inference
python run.py --mode inference --checkpoint outputs/checkpoints/best_model.pt
```

### Automated Data Loading

The pipeline automatically downloads the Kaggle dataset when needed:

```bash
# Download only
python run.py --mode download

# Or with auto-download during training
python run.py --mode train --auto-download

# Run entire pipeline (download → preprocess → train)
python run.py --mode all

# Force re-download
python run.py --mode download --force

# Use specific credentials
python run.py --mode download --kaggle-username YOUR_USER --kaggle-key YOUR_KEY
```

Data is downloaded from: [Multimodal Iris Fingerprint Biometric Data](https://www.kaggle.com/datasets/ninadmehendale/multimodal-iris-fingerprint-biometric-data)

### Databricks Deployment

```bash
# 1. Configure Databricks CLI
databricks configure

# 2. Validate bundle
databricks bundle validate -t dev

# 3. Deploy
databricks bundle deploy -t dev

# 4. Run training job
databricks bundle run biometric_training -t dev
```

## CI/CD Pipeline (Azure DevOps)

The project uses Azure DevOps for CI/CD:

| Trigger | Pipeline | Actions |
|---------|----------|--------|
| PR: `dev` → `main` | CI Pipeline | Lint, Unit Tests, Integration Tests, Deploy to `tests` |
| Push to `release` | Release Pipeline | Deploy to `prod`, Run Training & Inference |

### Branch Strategy
```
dev (development) ─PR─→ main (staging) ─merge─→ release (production)
        │                    │                         │
        │              CI Pipeline              Release Pipeline
        │           (tests + validation)        (prod deployment)
```

## Performance Analysis

### Data Loading Benchmarks
| Method | 1000 images | 10000 images | Memory |
|--------|-------------|--------------|--------|
| Sequential | 12.3s | 123s | Low |
| Ray (4 workers) | 3.5s | 35s | Medium |
| Ray (8 workers) | 2.1s | 21s | High |

### Bottlenecks & Mitigations
1. **I/O Bound**: Image loading → Mitigated with prefetching and Ray parallel
2. **Memory**: Large batches → Lazy loading, gradient accumulation
3. **GPU Util**: Data bottleneck → num_workers > 0 in DataLoader

## Technologies Used

| Category | Technology | Rationale |
|----------|------------|-----------|
| ML Framework | PyTorch | Industry standard, flexible |
| Parallel Processing | Ray | Easy scaling, fault tolerance |
| Data Format | PyArrow | Fast columnar operations |
| Config Management | YAML/Hydra-style | Human readable, versioned |
| Experiment Tracking | MLflow | Databricks integration |
| CI/CD | Azure DevOps | Enterprise integration |
| Cloud | Databricks/Azure | Enterprise ready |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License
MIT
