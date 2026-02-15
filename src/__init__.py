"""
Biometric MLOps Pipeline - Core Module

A production-quality machine learning infrastructure for multimodal 
biometric recognition with focus on scalability and MLOps best practices.
"""

__version__ = "1.0.0"
__author__ = "Biometric MLOps Team"

from .utils import Config, setup_logging, set_seed
from .data_loader import BiometricDataset, create_dataloaders
from .model import MultimodalBiometricModel
from .preprocessing import ParallelPreprocessor
from .data_download import download_dataset, check_kaggle_credentials
from .evaluate import ModelEvaluator, EvaluationMetrics
from .retrain import ChampionChallengerManager, RetrainingResult
from .monitoring import BiometricMonitor, DataDriftMonitor, DriftResult
from .model_registry import (
    UnityCatalogRegistry,
    ModelVersion,
    RegistrationResult,
    ModelAlias,
    get_model_registry,
)

__all__ = [
    # Core utilities
    "Config",
    "setup_logging", 
    "set_seed",
    # Data
    "BiometricDataset",
    "create_dataloaders",
    "download_dataset",
    "check_kaggle_credentials",
    # Model
    "MultimodalBiometricModel",
    "ParallelPreprocessor",
    # Evaluation
    "ModelEvaluator",
    "EvaluationMetrics",
    # Champion/Challenger
    "ChampionChallengerManager",
    "RetrainingResult",
    # Monitoring
    "BiometricMonitor",
    "DataDriftMonitor",
    "DriftResult",
    # Model Registry (Unity Catalog)
    "UnityCatalogRegistry",
    "ModelVersion",
    "RegistrationResult",
    "ModelAlias",
    "get_model_registry",
]
