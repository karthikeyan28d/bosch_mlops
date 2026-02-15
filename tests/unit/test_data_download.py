"""
Unit tests for data_download module.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_download import (
    check_kaggle_credentials,
    setup_kaggle_credentials,
    verify_and_organize_data,
    KAGGLE_DATASET,
)


class TestKaggleCredentials:
    """Test credential checking functions."""
    
    def test_check_credentials_with_env_vars(self):
        """Test credential check when env vars are set."""
        with patch.dict(os.environ, {
            "KAGGLE_USERNAME": "test_user",
            "KAGGLE_KEY": "test_key"
        }):
            assert check_kaggle_credentials() is True
    
    def test_check_credentials_without_env_vars(self):
        """Test credential check when no credentials available."""
        with patch.dict(os.environ, {}, clear=True):
            # Also need to mock the file check
            with patch("pathlib.Path.exists", return_value=False):
                assert check_kaggle_credentials() is False
    
    def test_setup_credentials_with_values(self):
        """Test setting up credentials programmatically."""
        # Clear any existing
        original_user = os.environ.pop("KAGGLE_USERNAME", None)
        original_key = os.environ.pop("KAGGLE_KEY", None)
        
        try:
            result = setup_kaggle_credentials("my_user", "my_key")
            assert result is True
            assert os.environ.get("KAGGLE_USERNAME") == "my_user"
            assert os.environ.get("KAGGLE_KEY") == "my_key"
        finally:
            # Restore
            if original_user:
                os.environ["KAGGLE_USERNAME"] = original_user
            if original_key:
                os.environ["KAGGLE_KEY"] = original_key


class TestDataVerification:
    """Test data verification and organization."""
    
    def test_verify_existing_data(self):
        """Test verification when data exists in correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create expected structure
            iris_dir = data_dir / "iris"
            fp_dir = data_dir / "fingerprint"
            iris_dir.mkdir()
            fp_dir.mkdir()
            
            # Create some test files
            (iris_dir / "001_001_1.bmp").write_text("test")
            (iris_dir / "001_001_2.bmp").write_text("test")
            (fp_dir / "001_001_1.BMP").write_text("test")
            (fp_dir / "001_001_2.BMP").write_text("test")
            
            result = verify_and_organize_data(data_dir)
            assert result is True
    
    def test_verify_empty_directory(self):
        """Test verification with empty data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = verify_and_organize_data(tmpdir)
            assert result is False
    
    def test_verify_nested_structure(self):
        """Test reorganization of nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create nested structure (as some Kaggle downloads do)
            nested = data_dir / "dataset-subfolder"
            nested.mkdir()
            iris_dir = nested / "iris"
            fp_dir = nested / "fingerprint"
            iris_dir.mkdir()
            fp_dir.mkdir()
            
            # Create test files
            (iris_dir / "test.bmp").write_text("test")
            (fp_dir / "test.BMP").write_text("test")
            
            result = verify_and_organize_data(data_dir)
            
            # Should have reorganized
            assert (data_dir / "iris").exists()
            assert (data_dir / "fingerprint").exists()
            assert result is True


class TestKaggleDataset:
    """Test Kaggle dataset configuration."""
    
    def test_dataset_identifier(self):
        """Verify correct dataset identifier."""
        assert KAGGLE_DATASET == "ninadmehendale/multimodal-iris-fingerprint-biometric-data"
    
    def test_download_skips_existing(self):
        """Test that download is skipped when data already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create existing data
            iris_dir = data_dir / "iris"
            fp_dir = data_dir / "fingerprint"
            iris_dir.mkdir()
            fp_dir.mkdir()
            (iris_dir / "test.bmp").write_text("test")
            (fp_dir / "test.BMP").write_text("test")
            
            # Import here to avoid issues if kaggle not installed
            from src.data_download import download_dataset
            
            # Should return True (data exists) without downloading
            result = download_dataset(output_dir=data_dir, force=False)
            assert result is True


class TestIntegrationWithConfig:
    """Test integration with config system."""
    
    def test_module_import(self):
        """Test that module can be imported."""
        from src import data_download
        assert hasattr(data_download, "download_dataset")
        assert hasattr(data_download, "main")
