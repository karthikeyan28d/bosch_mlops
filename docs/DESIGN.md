# System Design & Architecture

This document details the architectural decisions, design patterns, and trade-offs for the Multimodal Biometric MLOps Pipeline.

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Component Design](#component-design)
3. [Data Pipeline Architecture](#data-pipeline-architecture)
4. [ML Workflow Design](#ml-workflow-design)
5. [Scalability Analysis](#scalability-analysis)
6. [Trade-offs & Decisions](#trade-offs--decisions)
7. [Performance Bottlenecks](#performance-bottlenecks)
8. [Deployment Architecture](#deployment-architecture)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIOMETRIC MLOPS SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DATA LAYER                                   │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │  │  Raw Images  │───▶│  Preprocessor │───▶│   Feature    │          │    │
│  │  │  (Iris/FP)   │    │  (Ray/MP)     │    │    Store     │          │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         ML LAYER                                     │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │  │   Dataset    │───▶│   Training    │───▶│    Model     │          │    │
│  │  │   (PyTorch)  │    │    Loop       │    │   Registry   │          │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         SERVING LAYER                                │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │  │  Inference   │◀───│   Checkpoint  │    │  Predictions │          │    │
│  │  │   Pipeline   │    │    Loader     │───▶│   (Output)   │          │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         INFRASTRUCTURE LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Config     │  │   MLflow     │  │   CI/CD      │  │  Databricks  │   │
│  │   (YAML)     │  │   Tracking   │  │(Azure DevOps)│  │   Bundles    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Configuration Management (`src/utils.py`)

**Pattern**: Config-as-Code with hierarchical YAML

```
Config Loading:
  config.yaml ──┬──> Runtime Config
                │
  project_config.yml ──> Environment Overrides
```

**Key Design Decisions**:
- Single source of truth for all parameters
- Environment-specific overrides (dev/staging/prod)
- Type validation through structured access
- Supports both dot notation and bracket access

```python
# Access patterns
config.training.batch_size  # Dot notation
config["training"]["batch_size"]  # Bracket notation
config.get("optional", default)  # With defaults
```

### 2. Data Loading (`src/data_loader.py`)

**Pattern**: PyTorch Dataset abstraction with lazy loading

```
BiometricDataset
├── _discover_samples()     # File discovery
├── _build_label_mapping()  # Label encoding
├── _load_image()           # Lazy image loading
└── __getitem__()           # On-demand transform
```

**Multimodal Handling**:
- Paired iris-fingerprint samples by subject ID
- Consistent transforms applied to both modalities
- Label extracted from filename pattern (e.g., `001_L_01.bmp`)

**Memory Efficiency**:
```
Sequential Loading:  [Load] → [Transform] → [Return]
                     ↓
                     Only 1 sample in memory at a time
```

### 3. Preprocessing (`src/preprocessing.py`)

**Pattern**: Strategy pattern for parallel backends

```
ParallelPreprocessor
├── Ray backend        # Distributed processing
├── Multiprocessing    # Local parallelism
└── Sequential         # Debugging/small data
```

**Backend Selection Logic**:
```python
if backend == "ray" and RAY_AVAILABLE:
    self.backend = "ray"
elif backend == "multiprocessing":
    self.backend = "multiprocessing"
else:
    self.backend = "sequential"
```

**Performance Characteristics**:
| Backend | Best For | Overhead | Scalability |
|---------|----------|----------|-------------|
| Sequential | <100 samples | None | Single thread |
| Multiprocessing | 100-10K samples | Process spawn | CPU cores |
| Ray | >10K samples | Cluster init | Unlimited |

### 4. Model Architecture (`src/model.py`)

**Pattern**: Modular CNN with pluggable fusion

```
MultimodalBiometricModel
├── IrisBranch (ModalityBranch)
│   ├── ConvBlock × N
│   └── FC layers → embedding
├── FingerprintBranch (ModalityBranch)
│   ├── ConvBlock × N
│   └── FC layers → embedding
├── Fusion Layer
│   ├── Concat
│   ├── Attention
│   └── Weighted Sum
└── Classifier Head
```

**Fusion Strategies**:

1. **Concatenation** (Default)
   - Simple, no learnable parameters
   - Doubles feature dimension
   - Best baseline

2. **Attention Fusion**
   - Learns modality importance
   - Adaptive weighting per sample
   - More parameters

3. **Weighted Sum**
   - Learnable scalar weights
   - Maintains dimension
   - Lightweight

### 5. Training Pipeline (`src/train.py`)

**Pattern**: Trainer class encapsulating training loop

```
Trainer
├── train_epoch()      # Single epoch training
├── validate()         # Evaluation loop
├── save_checkpoint()  # Model persistence
├── load_checkpoint()  # Resume training
└── train()            # Full training loop
```

**Reproducibility Features**:
- Seed setting across all random sources
- Deterministic DataLoader shuffling
- Checkpoint includes config and optimizer state

**Training Flow**:
```
For each epoch:
    1. Train on all batches
    2. Validate
    3. Update scheduler
    4. Check early stopping
    5. Save checkpoint if best
    6. Log metrics
```

---

## Data Pipeline Architecture

### File-Based Flow

```
data/raw/
├── iris/
│   ├── 001_L_01.bmp
│   ├── 001_L_02.bmp
│   └── ...
└── fingerprint/
    ├── 001_L_thumb_01.BMP
    └── ...
        │
        ▼
┌──────────────────┐
│  BiometricDataset │
│  - Lazy loading   │
│  - On-fly transforms │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│    DataLoader    │
│  - Batching      │
│  - Shuffling     │
│  - Prefetching   │
└──────────────────┘
        │
        ▼
    Training
```

### Preprocessed Data Flow (Optional)

```
Raw Images
    │
    ▼ (Ray/MP parallel)
┌──────────────────┐
│  Preprocessing   │
│  - Resize        │
│  - Normalize     │
│  - Serialize     │
└──────────────────┘
    │
    ▼
Parquet/Arrow Cache
    │
    ▼
┌──────────────────┐
│  ProcessedDataset │
│  - Fast loading  │
│  - No transforms │
└──────────────────┘
```

---

## ML Workflow Design

### Training Workflow

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Data   │────▶│  Model  │────▶│ Metrics │
│ Loading │     │ Forward │     │  Loss   │
└─────────┘     └─────────┘     └─────────┘
                     │               │
                     ▼               ▼
              ┌─────────┐     ┌─────────┐
              │Backward │     │  MLflow │
              │  Pass   │     │   Log   │
              └─────────┘     └─────────┘
                     │
                     ▼
              ┌─────────┐
              │Optimizer│
              │  Step   │
              └─────────┘
```

### Inference Workflow

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Load    │────▶│ Model   │────▶│ Output  │
│Checkpoint│    │ Forward │     │ Export  │
└─────────┘     └─────────┘     └─────────┘
     │               │               │
     ▼               ▼               ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Config  │     │Embeddings│    │ Parquet │
│  Load   │     │(optional)│    │   CSV   │
└─────────┘     └─────────┘     └─────────┘
```

---

## Scalability Analysis

### Horizontal Scaling Points

| Component | Scaling Strategy | Limitation |
|-----------|------------------|------------|
| Preprocessing | Ray workers | Memory per worker |
| Data Loading | DataLoader workers | CPU cores |
| Training | Distributed PyTorch | GPU count |
| Inference | Batch parallelism | Memory |

### Memory Usage Patterns

```
Small Dataset (<1GB):
  - Load all to memory
  - Fast iteration
  - Simple debugging

Medium Dataset (1-10GB):
  - Lazy loading essential
  - Preprocessed cache helps
  - Memory-mapped files

Large Dataset (>10GB):
  - Streaming required
  - Distributed preprocessing
  - Sharded storage
```

### Throughput Benchmarks

| Operation | Sequential | 4 Workers | 8 Workers |
|-----------|------------|-----------|-----------|
| Load 1K images | 12.3s | 3.5s | 2.1s |
| Preprocess 1K | 15.2s | 4.1s | 2.5s |
| Training epoch | 45s | N/A (GPU bound) | N/A |

---

## Trade-offs & Decisions

### 1. PyTorch vs TensorFlow
**Decision**: PyTorch
- **Pro**: Dynamic graphs, debugging, research community
- **Con**: Slightly more boilerplate for production
- **Rationale**: Better for rapid iteration and custom architectures

### 2. Ray vs Multiprocessing
**Decision**: Both (configurable)
- **Ray**: Better for distributed, fault-tolerant
- **Multiprocessing**: No dependencies, simple
- **Rationale**: Allow user choice based on infrastructure

### 3. Parquet vs Raw Images
**Decision**: Support both
- **Raw**: No preprocessing overhead, flexible transforms
- **Parquet**: Fast loading, standardized features
- **Rationale**: Preprocessing is optional optimization

### 4. Single Model vs Ensemble
**Decision**: Single multimodal model
- **Pro**: Simpler, end-to-end trainable
- **Con**: Less flexibility for modality-specific tuning
- **Rationale**: Focus on system design, not model complexity

### 5. Local vs Cloud First
**Decision**: Local-first with cloud deployment
- **Pro**: Fast iteration, no cloud costs during dev
- **Con**: Need to ensure cloud compatibility
- **Rationale**: Developer experience priority

---

## Performance Bottlenecks

### Identified Bottlenecks

1. **Image Loading I/O**
   - **Impact**: 60% of data loading time
   - **Mitigation**: Preprocessing cache, prefetching
   - **Monitoring**: DataLoader iteration time

2. **Transform Computation**
   - **Impact**: 30% of data loading time
   - **Mitigation**: GPU transforms, preprocessing
   - **Monitoring**: CPU utilization

3. **GPU Memory**
   - **Impact**: Limits batch size
   - **Mitigation**: Gradient accumulation, mixed precision
   - **Monitoring**: CUDA memory stats

4. **Model Checkpoint Size**
   - **Impact**: Save/load time
   - **Mitigation**: Save only weights, use fp16
   - **Monitoring**: Checkpoint file size

### Monitoring Metrics

```python
metrics = {
    "data_load_time": "Time to load one batch",
    "forward_time": "Model forward pass time",
    "backward_time": "Gradient computation time",
    "gpu_memory_used": "Current GPU memory",
    "throughput": "Samples per second"
}
```

---

## Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────┐
│           Local Machine                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Python  │  │  MLflow │  │  Data   │ │
│  │  venv   │  │  Local  │  │  Local  │ │
│  └─────────┘  └─────────┘  └─────────┘ │
└─────────────────────────────────────────┘
```

### Databricks Deployment

```
┌─────────────────────────────────────────┐
│         Databricks Workspace             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │  Spark  │  │  MLflow │  │  Unity  │ │
│  │ Cluster │  │ Managed │  │ Catalog │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│       │            │            │       │
│       ▼            ▼            ▼       │
│  ┌─────────────────────────────────┐   │
│  │      Asset Bundles (DABs)        │   │
│  │  - Workflow definitions          │   │
│  │  - Job configurations            │   │
│  │  - Environment management        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### CI/CD Pipeline (Azure DevOps)

```
PR: dev → main
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Lint   │────▶│  Unit   │────▶│ Integ.  │
│  Check  │     │  Tests  │     │  Tests  │
└─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              ┌─────────┐
                              │ Bundle  │
                              │Validate │
                              └─────────┘
                                     │
                                     ▼
                              ┌─────────┐
                              │ Deploy  │
                              │ (tests) │
                              └─────────┘

Push to release
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Bundle  │────▶│ Deploy  │────▶│   Run   │
│Validate │     │ (prod)  │     │Training │
└─────────┘     └─────────┘     └─────────┘
```

---

## Future Improvements

1. **Distributed Training**: Add PyTorch DDP support
2. **Model Serving**: REST API with FastAPI
3. **Feature Store**: Integration with Feast
4. **Monitoring**: Prometheus metrics export
5. **A/B Testing**: Model version comparison framework
