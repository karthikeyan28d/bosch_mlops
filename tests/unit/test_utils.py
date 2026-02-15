"""
Unit tests for utility functions.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConfig:
    """Tests for Config class."""
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        from src.utils import Config
        
        # Create test config
        config_content = """
project:
  name: test_project
  version: "1.0.0"

training:
  batch_size: 32
  learning_rate: 0.001
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config.from_yaml(config_file)
        
        assert config.project.name == "test_project"
        assert config.training.batch_size == 32
        assert config.training.learning_rate == 0.001
    
    def test_config_bracket_access(self, tmp_path):
        """Test bracket notation access."""
        from src.utils import Config
        
        config_content = """
data:
  path: /data
  size: 100
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config.from_yaml(config_file)
        
        assert config["data"]["path"] == "/data"
        assert config["data"]["size"] == 100
    
    def test_config_get_with_default(self, tmp_path):
        """Test get method with default value."""
        from src.utils import Config
        
        config_content = """
setting: value
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config.from_yaml(config_file)
        
        assert config.get("setting") == "value"
        assert config.get("nonexistent", "default") == "default"
    
    def test_config_to_dict(self, tmp_path):
        """Test converting config to dict."""
        from src.utils import Config
        
        config_content = """
a: 1
b: 2
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config.from_yaml(config_file)
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert d["a"] == 1
        assert d["b"] == 2
    
    def test_config_missing_file_raises(self, tmp_path):
        """Test that missing config file raises error."""
        from src.utils import Config
        
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "nonexistent.yaml")


class TestSetSeed:
    """Tests for seed setting function."""
    
    def test_seed_reproducibility_torch(self):
        """Test that setting seed makes torch operations reproducible."""
        from src.utils import set_seed
        
        set_seed(42)
        a1 = torch.randn(10)
        
        set_seed(42)
        a2 = torch.randn(10)
        
        assert torch.allclose(a1, a2)
    
    def test_seed_reproducibility_numpy(self):
        """Test that setting seed makes numpy operations reproducible."""
        from src.utils import set_seed
        
        set_seed(42)
        a1 = np.random.randn(10)
        
        set_seed(42)
        a2 = np.random.randn(10)
        
        assert np.allclose(a1, a2)
    
    def test_different_seeds_different_values(self):
        """Test that different seeds produce different values."""
        from src.utils import set_seed
        
        set_seed(42)
        a1 = torch.randn(10)
        
        set_seed(123)
        a2 = torch.randn(10)
        
        assert not torch.allclose(a1, a2)


class TestGetDevice:
    """Tests for device detection."""
    
    def test_get_device_cpu(self):
        """Test explicit CPU device."""
        from src.utils import get_device
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_get_device_auto(self):
        """Test auto device detection."""
        from src.utils import get_device
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda"]


class TestEnsureDir:
    """Tests for directory creation."""
    
    def test_ensure_dir_creates_directory(self, tmp_path):
        """Test that directory is created."""
        from src.utils import ensure_dir
        
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()
        
        result = ensure_dir(new_dir)
        
        assert new_dir.exists()
        assert result == new_dir
    
    def test_ensure_dir_existing_directory(self, tmp_path):
        """Test with existing directory."""
        from src.utils import ensure_dir
        
        existing = tmp_path / "existing"
        existing.mkdir()
        
        result = ensure_dir(existing)
        
        assert existing.exists()
        assert result == existing


class TestAverageMeter:
    """Tests for AverageMeter class."""
    
    def test_average_meter_update(self):
        """Test meter updates correctly."""
        from src.utils import AverageMeter
        
        meter = AverageMeter("test")
        
        meter.update(10.0)
        meter.update(20.0)
        meter.update(30.0)
        
        assert meter.avg == 20.0
        assert meter.sum == 60.0
        assert meter.count == 3
    
    def test_average_meter_weighted_update(self):
        """Test weighted updates."""
        from src.utils import AverageMeter
        
        meter = AverageMeter()
        
        # 10.0 * 2 samples + 20.0 * 3 samples = 20 + 60 = 80
        # Total samples = 5
        # Average = 80 / 5 = 16
        meter.update(10.0, n=2)
        meter.update(20.0, n=3)
        
        assert meter.avg == 16.0
        assert meter.count == 5
    
    def test_average_meter_reset(self):
        """Test meter reset."""
        from src.utils import AverageMeter
        
        meter = AverageMeter()
        meter.update(100.0)
        
        meter.reset()
        
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0


class TestSetupLogging:
    """Tests for logging setup."""
    
    def test_setup_logging_level(self):
        """Test logging level is set correctly."""
        from src.utils import setup_logging
        
        logger = setup_logging(level="DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_file(self, tmp_path):
        """Test logging to file."""
        from src.utils import setup_logging
        
        log_file = tmp_path / "test.log"
        logger = setup_logging(level="INFO", log_file=log_file)
        
        logger.info("Test message")
        
        # Check file was created and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
