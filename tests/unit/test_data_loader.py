"""
Unit tests for data loading module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestBiometricDataset:
    """Tests for BiometricDataset class."""
    
    def test_dataset_initialization_empty_dir(self, tmp_path):
        """Test dataset handles empty directory gracefully."""
        from src.data_loader import BiometricDataset
        
        # Create empty directories
        (tmp_path / "iris").mkdir()
        (tmp_path / "fingerprint").mkdir()
        
        dataset = BiometricDataset(tmp_path)
        
        assert len(dataset) == 0
        assert dataset.num_classes == 0
    
    def test_dataset_sample_discovery(self, tmp_path):
        """Test that samples are discovered correctly."""
        from src.data_loader import BiometricDataset
        from PIL import Image
        
        # Create test images
        iris_dir = tmp_path / "iris"
        fp_dir = tmp_path / "fingerprint"
        iris_dir.mkdir()
        fp_dir.mkdir()
        
        # Create dummy images for two subjects
        for subject_id in ["001", "002"]:
            # Create iris image
            img = Image.new("L", (100, 100), color=128)
            img.save(iris_dir / f"{subject_id}_L_01.bmp")
            
            # Create fingerprint image
            img.save(fp_dir / f"{subject_id}_L_thumb_01.BMP")
        
        dataset = BiometricDataset(tmp_path)
        
        assert len(dataset) == 2
        assert dataset.num_classes == 2
    
    def test_dataset_getitem_returns_correct_structure(self, tmp_path):
        """Test that __getitem__ returns correct data structure."""
        from src.data_loader import BiometricDataset
        from PIL import Image
        
        # Setup
        iris_dir = tmp_path / "iris"
        fp_dir = tmp_path / "fingerprint"
        iris_dir.mkdir()
        fp_dir.mkdir()
        
        img = Image.new("L", (100, 100), color=128)
        img.save(iris_dir / "001_L_01.bmp")
        img.save(fp_dir / "001_L_thumb_01.BMP")
        
        dataset = BiometricDataset(tmp_path, image_size=(64, 64))
        
        sample = dataset[0]
        
        assert "iris" in sample
        assert "fingerprint" in sample
        assert "label" in sample
        assert "subject_id" in sample
        
        assert sample["iris"].shape == (1, 64, 64)
        assert sample["fingerprint"].shape == (1, 64, 64)
        assert isinstance(sample["label"], torch.Tensor)
    
    def test_dataset_transform_applied(self, tmp_path):
        """Test that transforms are correctly applied."""
        from src.data_loader import BiometricDataset
        from PIL import Image
        
        iris_dir = tmp_path / "iris"
        fp_dir = tmp_path / "fingerprint"
        iris_dir.mkdir()
        fp_dir.mkdir()
        
        img = Image.new("L", (200, 200), color=128)
        img.save(iris_dir / "001_L_01.bmp")
        img.save(fp_dir / "001_L_thumb_01.BMP")
        
        dataset = BiometricDataset(tmp_path, image_size=(32, 32))
        
        sample = dataset[0]
        
        # Check resize worked
        assert sample["iris"].shape == (1, 32, 32)
        
        # Check normalization (values should be around 0 for gray 128)
        assert sample["iris"].min() >= -1.0
        assert sample["iris"].max() <= 1.0


class TestCreateDataloaders:
    """Tests for dataloader creation."""
    
    def test_split_ratios(self, tmp_path):
        """Test that split ratios are respected."""
        from src.data_loader import BiometricDataset, create_dataloaders
        from PIL import Image
        
        # Create 10 subjects
        iris_dir = tmp_path / "iris"
        fp_dir = tmp_path / "fingerprint"
        iris_dir.mkdir()
        fp_dir.mkdir()
        
        for i in range(10):
            subject_id = f"{i:03d}"
            img = Image.new("L", (64, 64), color=128)
            img.save(iris_dir / f"{subject_id}_L_01.bmp")
            img.save(fp_dir / f"{subject_id}_L_thumb_01.BMP")
        
        train_loader, val_loader, test_loader, num_classes = create_dataloaders(
            tmp_path,
            batch_size=2,
            split_ratios=(0.6, 0.2, 0.2),
            num_workers=0
        )
        
        assert num_classes == 10
        
        # Check approximate split sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        
        assert train_size == 6
        assert val_size == 2
        assert test_size == 2


class TestCollateMultimodal:
    """Tests for custom collate function."""
    
    def test_collate_stacks_correctly(self):
        """Test that collate function stacks tensors correctly."""
        from src.data_loader import collate_multimodal
        
        batch = [
            {
                "iris": torch.randn(1, 64, 64),
                "fingerprint": torch.randn(1, 64, 64),
                "label": torch.tensor(0),
                "subject_id": "001"
            },
            {
                "iris": torch.randn(1, 64, 64),
                "fingerprint": torch.randn(1, 64, 64),
                "label": torch.tensor(1),
                "subject_id": "002"
            }
        ]
        
        collated = collate_multimodal(batch)
        
        assert collated["iris"].shape == (2, 1, 64, 64)
        assert collated["fingerprint"].shape == (2, 1, 64, 64)
        assert collated["label"].shape == (2,)
        assert len(collated["subject_id"]) == 2
