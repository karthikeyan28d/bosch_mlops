"""
Utility functions and configuration management.

This module provides:
- Config: YAML-based configuration management
- Logging setup
- Seed management for reproducibility
- Device detection
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

import yaml
import numpy as np
import torch


@dataclass
class Config:
    """
    Configuration manager supporting YAML files with environment overrides.
    
    Supports:
    - Loading from YAML files
    - Environment variable overrides
    - Nested attribute access
    - Default values
    
    Example:
        config = Config.from_yaml("configs/config.yaml")
        batch_size = config.training.batch_size
        learning_rate = config["training"]["learning_rate"]
    """
    _data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path], env: str = "dev") -> "Config":
        """
        Load configuration from YAML file with optional environment overrides.
        
        Args:
            config_path: Path to the main config file
            env: Environment name for env-specific overrides
            
        Returns:
            Config instance with loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Apply environment overrides if project_config.yml exists
        project_config_path = config_path.parent.parent / "project_config.yml"
        if project_config_path.exists():
            with open(project_config_path, "r", encoding="utf-8") as f:
                project_data = yaml.safe_load(f)
                # Merge environment-specific config
                if env in project_data:
                    data["environment_config"] = project_data[env]
                # Merge shared project config
                for key, value in project_data.items():
                    if not isinstance(value, dict) or key not in ["dev", "prod"]:
                        data[key] = value
        
        return cls(_data=data)
    
    def __getattr__(self, name: str) -> Any:
        """Enable dot notation access to config values."""
        if name == "_data":
            return object.__getattribute__(self, "_data")
        
        data = object.__getattribute__(self, "_data")
        if name in data:
            value = data[name]
            if isinstance(value, dict):
                return Config(_data=value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Enable bracket notation access to config values."""
        return self._data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._data.copy()
    
    def __repr__(self) -> str:
        return f"Config({self._data})"


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Custom log format string
        log_file: Optional file path for log output
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger("biometric_mlops")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_config: str = "auto") -> torch.device:
    """
    Determine the device to use for training/inference.
    
    Args:
        device_config: Device configuration ("auto", "cuda", "cpu")
        
    Returns:
        torch.device instance
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_config)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_databricks() -> bool:
    """Check if running in Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_dbutils():
    """
    Get Databricks dbutils if available.
    
    Returns:
        dbutils if in Databricks, None otherwise
    """
    if is_databricks():
        try:
            from pyspark.dbutils import DBUtils
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            return DBUtils(spark)
        except ImportError:
            return None
    return None


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
