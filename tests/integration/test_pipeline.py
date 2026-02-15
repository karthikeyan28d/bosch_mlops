"""
Integration tests for the complete pipeline.

These tests verify end-to-end functionality of the ML pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample dataset for testing."""
    iris_dir = tmp_path / "iris"
    fp_dir = tmp_path / "fingerprint"
    iris_dir.mkdir()
    fp_dir.mkdir()
    
    # Create 5 subjects with 2 samples each
    for subject_id in range(5):
        for sample_idx in range(2):
            subject_str = f"{subject_id:03d}"
            
            # Create gray images with different intensities per subject
            intensity = 50 + subject_id * 40  # Different intensity per subject
            
            iris_img = Image.new("L", (128, 128), color=intensity)
            fp_img = Image.new("L", (128, 128), color=intensity)
            
            iris_img.save(iris_dir / f"{subject_str}_L_{sample_idx:02d}.bmp")
            fp_img.save(fp_dir / f"{subject_str}_L_thumb_{sample_idx:02d}.BMP")
    
    return tmp_path


@pytest.fixture
def config_file(tmp_path):
    """Create a test config file."""
    config_content = """
project:
  name: "test_biometric"
  version: "1.0.0"

runtime:
  environment: "local"
  seed: 42
  device: "cpu"
  num_workers: 0
  debug: false

data:
  raw_dir: "{data_dir}"
  processed_dir: "data/processed"
  output_dir: "outputs"
  image:
    size: [64, 64]
    mean: [0.5]
    std: [0.5]
  split:
    train: 0.6
    val: 0.2
    test: 0.2

preprocessing:
  backend: "sequential"
  num_workers: 1
  batch_size: 10
  cache_enabled: false

model:
  architecture: "multimodal_cnn"
  num_classes: null
  embedding_dim: 32
  dropout: 0.1
  fusion_method: "concat"
  iris_branch:
    conv_layers:
      - [16, 3, 1]
      - [32, 3, 1]
  fingerprint_branch:
    conv_layers:
      - [16, 3, 1]
      - [32, 3, 1]

training:
  epochs: 2
  batch_size: 4
  learning_rate: 0.01
  weight_decay: 0.0001
  optimizer: "adam"
  scheduler: "none"
  scheduler_params:
    T_max: 2
  early_stopping:
    enabled: false
    patience: 5
    min_delta: 0.001
    monitor: "val_loss"
  checkpoint:
    enabled: true
    save_dir: "{output_dir}/checkpoints"
    save_best_only: true
    save_freq: 1
  gradient_clip: 1.0
  mixed_precision: false

inference:
  batch_size: 8
  checkpoint: "latest"
  output_format: "csv"
  output_path: "{output_dir}/predictions.csv"

mlflow:
  enabled: false

logging:
  level: "INFO"
  format: "%(message)s"
  file_path: null
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    def test_data_loading_pipeline(self, sample_dataset):
        """Test complete data loading pipeline."""
        from src.data_loader import BiometricDataset, create_dataloaders
        
        # Create dataset
        dataset = BiometricDataset(sample_dataset, image_size=(64, 64))
        
        assert len(dataset) == 10  # 5 subjects * 2 samples
        assert dataset.num_classes == 5
        
        # Create dataloaders
        train_loader, val_loader, test_loader, num_classes = create_dataloaders(
            sample_dataset,
            batch_size=4,
            num_workers=0,
            split_ratios=(0.6, 0.2, 0.2),
            image_size=(64, 64)
        )
        
        assert num_classes == 5
        
        # Test iteration
        for batch in train_loader:
            assert "iris" in batch
            assert "fingerprint" in batch
            assert "label" in batch
            break
    
    def test_model_training_loop(self, sample_dataset):
        """Test that a training step works correctly."""
        from src.data_loader import create_dataloaders
        from src.model import MultimodalBiometricModel
        
        # Setup
        train_loader, _, _, num_classes = create_dataloaders(
            sample_dataset,
            batch_size=4,
            num_workers=0,
            image_size=(64, 64)
        )
        
        model = MultimodalBiometricModel(
            num_classes=num_classes,
            embedding_dim=32,
            conv_channels=[16, 32]
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Run one training step
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            output = model(batch["iris"], batch["fingerprint"])
            loss = criterion(output["logits"], batch["label"])
            
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0
            assert torch.isfinite(loss)
            break
    
    def test_model_inference(self, sample_dataset):
        """Test inference pipeline."""
        from src.data_loader import BiometricDataset
        from src.model import MultimodalBiometricModel
        from torch.utils.data import DataLoader
        
        dataset = BiometricDataset(sample_dataset, image_size=(64, 64))
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        model = MultimodalBiometricModel(
            num_classes=dataset.num_classes,
            embedding_dim=32,
            conv_channels=[16, 32]
        )
        model.eval()
        
        # Run inference
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                output = model(batch["iris"], batch["fingerprint"])
                preds = output["logits"].argmax(dim=1)
                all_preds.extend(preds.tolist())
        
        assert len(all_preds) == len(dataset)
    
    def test_checkpoint_save_load(self, sample_dataset, tmp_path):
        """Test checkpoint saving and loading."""
        from src.model import MultimodalBiometricModel
        
        # Create and train model briefly
        model1 = MultimodalBiometricModel(num_classes=5, embedding_dim=32)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({
            "epoch": 1,
            "model_state_dict": model1.state_dict(),
            "config": {"model": {"num_classes": 5, "embedding_dim": 32}}
        }, checkpoint_path)
        
        # Load into new model
        model2 = MultimodalBiometricModel(num_classes=5, embedding_dim=32)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint["model_state_dict"])
        
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model1.state_dict().items(),
            model2.state_dict().items()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline."""
    
    def test_sequential_preprocessing(self, sample_dataset):
        """Test sequential preprocessing."""
        from src.preprocessing import ParallelPreprocessor
        from src.data_loader import BiometricDataset
        
        dataset = BiometricDataset(sample_dataset)
        
        preprocessor = ParallelPreprocessor(
            backend="sequential",
            target_size=(64, 64)
        )
        
        processed, stats = preprocessor.process(dataset.samples)
        
        assert stats.processed_samples == len(dataset.samples)
        assert stats.failed_samples == 0
        assert stats.throughput > 0
    
    def test_multiprocessing_preprocessing(self, sample_dataset):
        """Test multiprocessing preprocessing."""
        from src.preprocessing import ParallelPreprocessor
        from src.data_loader import BiometricDataset
        
        dataset = BiometricDataset(sample_dataset)
        
        preprocessor = ParallelPreprocessor(
            backend="multiprocessing",
            num_workers=2,
            target_size=(64, 64)
        )
        
        processed, stats = preprocessor.process(dataset.samples)
        
        assert stats.processed_samples == len(dataset.samples)


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_seed_reproducibility(self, sample_dataset):
        """Test that setting seed produces reproducible results."""
        from src.utils import set_seed
        from src.data_loader import create_dataloaders
        from src.model import MultimodalBiometricModel
        
        def run_training_step():
            set_seed(42)
            
            train_loader, _, _, num_classes = create_dataloaders(
                sample_dataset,
                batch_size=4,
                num_workers=0,
                image_size=(64, 64),
                seed=42
            )
            
            model = MultimodalBiometricModel(
                num_classes=num_classes,
                embedding_dim=32,
                conv_channels=[16, 32]
            )
            
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch["iris"], batch["fingerprint"])
                loss = criterion(output["logits"], batch["label"])
                loss.backward()
                optimizer.step()
                return loss.item()
        
        # Run twice with same seed
        loss1 = run_training_step()
        loss2 = run_training_step()
        
        assert abs(loss1 - loss2) < 1e-5
