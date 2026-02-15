"""
Model Performance Evaluation Module.

Provides comprehensive model evaluation including:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Per-class performance
- Model comparison utilities
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .utils import Config, setup_logging, set_seed, get_device
from .data_loader import create_dataloaders
from .model import MultimodalBiometricModel
from .model_registry import get_model_registry, ModelAlias

logger = logging.getLogger("biometric_mlops")


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    num_classes: int
    num_samples: int
    inference_time_ms: float
    timestamp: str
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "EvaluationMetrics":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class ModelEvaluator:
    """
    Comprehensive model evaluator.
    
    Evaluates model performance on test data and generates
    detailed metrics for comparison and monitoring.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_path: Optional[str] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate model on given dataloader.
        
        Args:
            dataloader: Test data loader
            model_path: Path to model checkpoint (for logging)
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for evaluation. Install: pip install scikit-learn")
        
        all_preds = []
        all_labels = []
        total_time = 0.0
        
        for batch in dataloader:
            iris = batch["iris"].to(self.device)
            fingerprint = batch["fingerprint"].to(self.device)
            labels = batch["label"]
            
            # Time inference
            start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            
            if start:
                start.record()
            
            import time
            cpu_start = time.perf_counter()
            
            outputs = self.model(iris, fingerprint)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            if start:
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
            else:
                total_time += (time.perf_counter() - cpu_start) * 1000
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = EvaluationMetrics(
            accuracy=accuracy_score(all_labels, all_preds),
            precision_macro=precision_score(all_labels, all_preds, average="macro", zero_division=0),
            recall_macro=recall_score(all_labels, all_preds, average="macro", zero_division=0),
            f1_macro=f1_score(all_labels, all_preds, average="macro", zero_division=0),
            precision_weighted=precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            recall_weighted=recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            f1_weighted=f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            num_classes=self.num_classes,
            num_samples=len(all_labels),
            inference_time_ms=total_time / len(dataloader),
            timestamp=datetime.now().isoformat(),
            model_path=model_path,
        )
        
        return metrics
    
    def get_confusion_matrix(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> np.ndarray:
        """Get confusion matrix for predictions."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                iris = batch["iris"].to(self.device)
                fingerprint = batch["fingerprint"].to(self.device)
                labels = batch["label"]
                
                outputs = self.model(iris, fingerprint)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return confusion_matrix(all_labels, all_preds)
    
    def get_classification_report(
        self,
        dataloader: torch.utils.data.DataLoader,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """Get detailed classification report."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                iris = batch["iris"].to(self.device)
                fingerprint = batch["fingerprint"].to(self.device)
                labels = batch["label"]
                
                outputs = self.model(iris, fingerprint)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return classification_report(
            all_labels, all_preds,
            target_names=class_names,
            zero_division=0
        )


def compare_models(
    metrics_a: EvaluationMetrics,
    metrics_b: EvaluationMetrics,
    primary_metric: str = "f1_macro",
) -> Tuple[str, float]:
    """
    Compare two models and determine which is better.
    
    Args:
        metrics_a: Metrics for model A (typically champion)
        metrics_b: Metrics for model B (typically challenger)
        primary_metric: Metric to use for comparison
        
    Returns:
        Tuple of (winner: "A" or "B", improvement percentage)
    """
    value_a = getattr(metrics_a, primary_metric)
    value_b = getattr(metrics_b, primary_metric)
    
    if value_b > value_a:
        improvement = ((value_b - value_a) / value_a) * 100 if value_a > 0 else float("inf")
        return "B", improvement
    else:
        improvement = ((value_a - value_b) / value_b) * 100 if value_b > 0 else 0
        return "A", -improvement


def evaluate_checkpoint(
    checkpoint_path: str,
    config: Config,
    split: str = "test",
) -> EvaluationMetrics:
    """
    Evaluate a model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration object
        split: Data split to evaluate on
        
    Returns:
        EvaluationMetrics
    """
    device = get_device(config.runtime.device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", config.model.num_classes)
    
    # Create model
    model = MultimodalBiometricModel(
        num_classes=num_classes,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        fusion_method=config.model.get("fusion_method", "attention"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create dataloader
    dataloaders = create_dataloaders(
        data_dir=config.data.raw_dir,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers,
        image_size=tuple(config.data.image.size),
        seed=config.runtime.seed,
    )
    
    dataloader = dataloaders.get(split, dataloaders["test"])
    
    # Evaluate
    evaluator = ModelEvaluator(model, device, num_classes)
    metrics = evaluator.evaluate(dataloader, model_path=checkpoint_path)
    
    return metrics


def evaluate_from_registry(
    config: Config,
    alias: str = "champion",
    version: Optional[int] = None,
    split: str = "test",
) -> EvaluationMetrics:
    """
    Evaluate a model loaded from Unity Catalog.
    
    Args:
        config: Configuration object
        alias: Model alias (champion, challenger)
        version: Specific version to load (overrides alias)
        split: Data split to evaluate on
        
    Returns:
        EvaluationMetrics
    """
    device = get_device(config.runtime.device)
    
    # Get model from registry
    registry = get_model_registry(config)
    
    if version:
        model = registry.load_model(version=version)
        model_identifier = f"version {version}"
    else:
        model = registry.load_model(alias=alias)
        model_identifier = f"alias '{alias}'"
    
    logger.info(f"Loaded model from registry ({model_identifier})")
    
    # Create dataloader
    dataloaders = create_dataloaders(
        data_dir=config.data.raw_dir,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers,
        image_size=tuple(config.data.image.size),
        seed=config.runtime.seed,
    )
    
    dataloader = dataloaders.get(split, dataloaders["test"])
    
    # Evaluate
    evaluator = ModelEvaluator(model, device, model.num_classes)
    metrics = evaluator.evaluate(dataloader, model_path=f"registry:{model_identifier}")
    
    return metrics


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--config", "-c", default="configs/config.yaml")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--alias", choices=["champion", "challenger"], 
                        help="Load model from registry by alias")
    parser.add_argument("--version", type=int, help="Load specific version from registry")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", "-o", help="Output path for metrics JSON")
    parser.add_argument("--compare-with", help="Compare with another checkpoint")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.checkpoint and not args.alias and not args.version:
        parser.error("Must specify --checkpoint, --alias, or --version")
    
    setup_logging()
    config = Config.from_yaml(args.config)
    set_seed(config.runtime.seed)
    
    # Evaluate based on source
    if args.checkpoint:
        logger.info(f"Evaluating checkpoint: {args.checkpoint}")
        metrics = evaluate_checkpoint(args.checkpoint, config, args.split)
        source = args.checkpoint
    else:
        alias = args.alias or "champion"
        logger.info(f"Evaluating from registry (alias={alias}, version={args.version})")
        metrics = evaluate_from_registry(config, alias, args.version, args.split)
        source = f"registry:{alias}" if not args.version else f"registry:v{args.version}"
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Source: {source}")
    print(f"Split: {args.split}")
    print(f"Samples: {metrics.num_samples}")
    print(f"\nMetrics:")
    print(f"  Accuracy:         {metrics.accuracy:.4f}")
    print(f"  F1 (macro):       {metrics.f1_macro:.4f}")
    print(f"  F1 (weighted):    {metrics.f1_weighted:.4f}")
    print(f"  Precision (macro):{metrics.precision_macro:.4f}")
    print(f"  Recall (macro):   {metrics.recall_macro:.4f}")
    print(f"  Inference time:   {metrics.inference_time_ms:.2f}ms/batch")
    
    # Save metrics
    if args.output:
        metrics.to_json(args.output)
        print(f"\nMetrics saved to: {args.output}")
    
    # Compare with another model
    if args.compare_with:
        logger.info(f"Comparing with: {args.compare_with}")
        other_metrics = evaluate_checkpoint(args.compare_with, config, args.split)
        
        winner, improvement = compare_models(metrics, other_metrics)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"Model A (Champion): {source}")
        print(f"Model B (Challenger): {args.compare_with}")
        print(f"\nWinner: Model {winner}")
        print(f"F1 Improvement: {improvement:+.2f}%")
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE and config.mlflow.enabled:
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(metrics.to_dict())
            if args.output:
                mlflow.log_artifact(args.output)


if __name__ == "__main__":
    main()
