"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield


@pytest.fixture
def sample_images(tmp_path):
    """Create sample biometric images for testing."""
    iris_dir = tmp_path / "iris"
    fp_dir = tmp_path / "fingerprint"
    iris_dir.mkdir()
    fp_dir.mkdir()
    
    # Create a few test images
    for i in range(3):
        subject_id = f"{i:03d}"
        
        iris_img = Image.new("L", (128, 128), color=100 + i * 50)
        fp_img = Image.new("L", (128, 128), color=100 + i * 50)
        
        iris_img.save(iris_dir / f"{subject_id}_L_01.bmp")
        fp_img.save(fp_dir / f"{subject_id}_L_thumb_01.BMP")
    
    return tmp_path


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    from src.utils import Config
    
    return Config(_data={
        "runtime": {
            "seed": 42,
            "device": "cpu",
            "num_workers": 0
        },
        "data": {
            "raw_dir": "data/raw",
            "image": {
                "size": [64, 64]
            },
            "split": {
                "train": 0.7,
                "val": 0.15,
                "test": 0.15
            }
        },
        "model": {
            "embedding_dim": 64,
            "dropout": 0.3,
            "fusion_method": "concat"
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adam",
            "scheduler": "none",
            "gradient_clip": 1.0,
            "mixed_precision": False,
            "early_stopping": {
                "enabled": False,
                "patience": 5,
                "min_delta": 0.001
            },
            "checkpoint": {
                "enabled": True,
                "save_dir": "outputs/checkpoints",
                "save_best_only": True
            },
            "scheduler_params": {
                "T_max": 2
            }
        },
        "inference": {
            "batch_size": 8,
            "output_format": "csv"
        },
        "mlflow": {
            "enabled": False
        },
        "logging": {
            "level": "WARNING"
        }
    })
