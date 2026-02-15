"""
Champion/Challenger Model Retraining Module.

Implements the champion/challenger pattern for model retraining:
- Champion: Current production model
- Challenger: Newly trained model

The challenger only replaces the champion if it performs better
on the evaluation dataset, ensuring production stability.
"""

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .utils import Config, setup_logging, set_seed, get_device
from .data_loader import create_dataloaders
from .model import MultimodalBiometricModel
from .train import Trainer
from .evaluate import ModelEvaluator, EvaluationMetrics, compare_models
from .model_registry import get_model_registry, UnityCatalogRegistry

logger = logging.getLogger("biometric_mlops")


# Model registry paths
DEFAULT_CHAMPION_PATH = "outputs/models/champion"
DEFAULT_CHALLENGER_PATH = "outputs/models/challenger"
DEFAULT_ARCHIVE_PATH = "outputs/models/archive"


@dataclass
class RetrainingResult:
    """Result of retraining process."""
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    winner: str  # "champion" or "challenger"
    promoted: bool  # Whether challenger was promoted
    improvement: float  # Percentage improvement
    primary_metric: str
    threshold: float
    timestamp: str
    champion_path: str
    challenger_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ChampionChallengerManager:
    """
    Manages champion/challenger model lifecycle.
    
    Key features:
    - Track champion model in production
    - Train challenger models
    - Compare performance fairly
    - Promote challenger only if significantly better
    - Archive old champions for rollback
    """
    
    def __init__(
        self,
        config: Config,
        champion_dir: str = DEFAULT_CHAMPION_PATH,
        challenger_dir: str = DEFAULT_CHALLENGER_PATH,
        archive_dir: str = DEFAULT_ARCHIVE_PATH,
        promotion_threshold: float = 0.01,  # 1% improvement required
        primary_metric: str = "f1_macro",
        use_registry: bool = True,
    ):
        """
        Initialize manager.
        
        Args:
            config: Configuration object
            champion_dir: Directory for champion model
            challenger_dir: Directory for challenger model
            archive_dir: Directory for archived models
            promotion_threshold: Minimum improvement to promote (0.01 = 1%)
            primary_metric: Metric to compare models on
            use_registry: Whether to use Unity Catalog registry
        """
        self.config = config
        self.champion_dir = Path(champion_dir)
        self.challenger_dir = Path(challenger_dir)
        self.archive_dir = Path(archive_dir)
        self.promotion_threshold = promotion_threshold
        self.primary_metric = primary_metric
        self.use_registry = use_registry
        
        # Create directories
        self.champion_dir.mkdir(parents=True, exist_ok=True)
        self.challenger_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = get_device(config.runtime.device)
        
        # Initialize registry if enabled
        self.registry = None
        if use_registry:
            try:
                registry_config = getattr(config, "model_registry", None)
                if registry_config and getattr(registry_config, "enabled", False):
                    self.registry = get_model_registry(config)
                    logger.info("Unity Catalog registry enabled for retraining")
            except Exception as e:
                logger.warning(f"Could not initialize registry: {e}")
    
    @property
    def champion_checkpoint(self) -> Optional[Path]:
        """Get path to champion checkpoint."""
        ckpt = self.champion_dir / "model.pt"
        return ckpt if ckpt.exists() else None
    
    @property
    def champion_metrics_path(self) -> Optional[Path]:
        """Get path to champion metrics."""
        path = self.champion_dir / "metrics.json"
        return path if path.exists() else None
    
    def get_champion_metrics(self) -> Optional[EvaluationMetrics]:
        """Load champion metrics if available."""
        if self.champion_metrics_path:
            return EvaluationMetrics.from_json(str(self.champion_metrics_path))
        return None
    
    def train_challenger(
        self,
        train_loader,
        val_loader,
        num_classes: int,
    ) -> Tuple[str, EvaluationMetrics]:
        """
        Train a new challenger model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of output classes
            
        Returns:
            Tuple of (checkpoint_path, metrics)
        """
        logger.info("Training challenger model...")
        
        # Create model
        model = MultimodalBiometricModel(
            num_classes=num_classes,
            embedding_dim=self.config.model.embedding_dim,
            dropout=self.config.model.dropout,
            fusion_method=self.config.model.get("fusion_method", "attention"),
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=self.config,
            device=self.device,
            checkpoint_dir=str(self.challenger_dir),
        )
        
        # Train
        trainer.train(train_loader, val_loader, epochs=self.config.training.epochs)
        
        # Get best checkpoint
        challenger_ckpt = self.challenger_dir / "best_model.pt"
        
        # Copy to standard name
        final_ckpt = self.challenger_dir / "model.pt"
        if challenger_ckpt.exists():
            shutil.copy(challenger_ckpt, final_ckpt)
        
        # Evaluate challenger
        evaluator = ModelEvaluator(model, self.device, num_classes)
        metrics = evaluator.evaluate(val_loader, model_path=str(final_ckpt))
        
        # Save metrics
        metrics.to_json(str(self.challenger_dir / "metrics.json"))
        
        return str(final_ckpt), metrics
    
    def evaluate_model(
        self,
        checkpoint_path: str,
        test_loader,
        num_classes: int,
    ) -> EvaluationMetrics:
        """
        Evaluate a model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            test_loader: Test data loader
            num_classes: Number of classes
            
        Returns:
            EvaluationMetrics
        """
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = MultimodalBiometricModel(
            num_classes=num_classes,
            embedding_dim=self.config.model.embedding_dim,
            dropout=self.config.model.dropout,
            fusion_method=self.config.model.get("fusion_method", "attention"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Evaluate
        evaluator = ModelEvaluator(model, self.device, num_classes)
        return evaluator.evaluate(test_loader, model_path=checkpoint_path)
    
    def compare_and_promote(
        self,
        challenger_metrics: EvaluationMetrics,
        test_loader=None,
        num_classes: int = None,
    ) -> RetrainingResult:
        """
        Compare challenger with champion and promote if better.
        
        Args:
            challenger_metrics: Metrics from challenger evaluation
            test_loader: Optional test loader for re-evaluation
            num_classes: Number of classes (required if test_loader provided)
            
        Returns:
            RetrainingResult with comparison details
        """
        champion_metrics = self.get_champion_metrics()
        
        # If no champion exists, promote challenger automatically
        if champion_metrics is None:
            logger.info("No champion found. Promoting challenger as initial champion.")
            self._promote_challenger()
            
            return RetrainingResult(
                champion_metrics={},
                challenger_metrics=challenger_metrics.to_dict(),
                winner="challenger",
                promoted=True,
                improvement=float("inf"),
                primary_metric=self.primary_metric,
                threshold=self.promotion_threshold,
                timestamp=datetime.now().isoformat(),
                champion_path=str(self.champion_dir / "model.pt"),
                challenger_path=str(self.challenger_dir / "model.pt"),
            )
        
        # Re-evaluate champion on same test set if provided
        if test_loader is not None and num_classes is not None:
            logger.info("Re-evaluating champion on test set...")
            champion_metrics = self.evaluate_model(
                str(self.champion_checkpoint),
                test_loader,
                num_classes,
            )
        
        # Compare models
        champion_value = getattr(champion_metrics, self.primary_metric)
        challenger_value = getattr(challenger_metrics, self.primary_metric)
        
        improvement = (challenger_value - champion_value) / champion_value if champion_value > 0 else float("inf")
        
        # Determine winner
        if improvement > self.promotion_threshold:
            winner = "challenger"
            promoted = True
            logger.info(
                f"Challenger wins! {self.primary_metric}: "
                f"{champion_value:.4f} -> {challenger_value:.4f} "
                f"(+{improvement*100:.2f}%)"
            )
            self._promote_challenger()
        else:
            winner = "champion"
            promoted = False
            logger.info(
                f"Champion defended. {self.primary_metric}: "
                f"champion={champion_value:.4f}, challenger={challenger_value:.4f} "
                f"(improvement {improvement*100:.2f}% < threshold {self.promotion_threshold*100:.1f}%)"
            )
        
        result = RetrainingResult(
            champion_metrics=champion_metrics.to_dict(),
            challenger_metrics=challenger_metrics.to_dict(),
            winner=winner,
            promoted=promoted,
            improvement=improvement * 100,
            primary_metric=self.primary_metric,
            threshold=self.promotion_threshold,
            timestamp=datetime.now().isoformat(),
            champion_path=str(self.champion_dir / "model.pt"),
            challenger_path=str(self.challenger_dir / "model.pt"),
        )
        
        # Save result
        result.to_json(str(self.challenger_dir / "comparison_result.json"))
        
        return result
    
    def _promote_challenger(self) -> None:
        """Promote challenger to champion, archiving current champion."""
        # Archive current champion if exists
        if self.champion_checkpoint and self.champion_checkpoint.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.archive_dir / f"champion_{timestamp}"
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all files
            for f in self.champion_dir.iterdir():
                if f.is_file():
                    shutil.copy(f, archive_path / f.name)
            
            logger.info(f"Archived previous champion to: {archive_path}")
        
        # Clear champion directory
        for f in self.champion_dir.iterdir():
            if f.is_file():
                f.unlink()
        
        # Copy challenger to champion
        for f in self.challenger_dir.iterdir():
            if f.is_file() and not f.name.startswith("comparison"):
                shutil.copy(f, self.champion_dir / f.name)
        
        logger.info("Challenger promoted to champion!")
        
        # Update registry if available
        if self.registry:
            try:
                self.registry.promote_challenger()
                logger.info("Registry updated: challenger promoted to champion alias")
            except Exception as e:
                logger.warning(f"Could not update registry: {e}")
    
    def rollback(self, archive_name: Optional[str] = None) -> bool:
        """
        Rollback to a previous champion.
        
        Args:
            archive_name: Specific archive to restore (default: latest)
            
        Returns:
            True if rollback successful
        """
        archives = sorted(self.archive_dir.iterdir(), reverse=True)
        
        if not archives:
            logger.error("No archived models available for rollback")
            return False
        
        # Select archive
        if archive_name:
            archive_path = self.archive_dir / archive_name
            if not archive_path.exists():
                logger.error(f"Archive not found: {archive_name}")
                return False
        else:
            archive_path = archives[0]
        
        # Clear current champion
        for f in self.champion_dir.iterdir():
            if f.is_file():
                f.unlink()
        
        # Restore from archive
        for f in archive_path.iterdir():
            if f.is_file():
                shutil.copy(f, self.champion_dir / f.name)
        
        logger.info(f"Rolled back to: {archive_path}")
        
        # Update registry if available
        if self.registry:
            try:
                # Rollback in registry (if there's a previous champion version)
                self.registry.rollback()
                logger.info("Registry updated: rolled back to previous champion version")
            except Exception as e:
                logger.warning(f"Could not rollback in registry: {e}")
        
        return True


def retrain_pipeline(
    config: Config,
    force_promote: bool = False,
    promotion_threshold: float = 0.01,
) -> RetrainingResult:
    """
    Run complete retraining pipeline with champion/challenger comparison.
    
    Args:
        config: Configuration object
        force_promote: Force promote challenger regardless of performance
        promotion_threshold: Minimum improvement required for promotion
        
    Returns:
        RetrainingResult
    """
    set_seed(config.runtime.seed)
    
    # Create data loaders
    dataloaders = create_dataloaders(
        data_dir=config.data.raw_dir,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers,
        image_size=tuple(config.data.image.size),
        seed=config.runtime.seed,
    )
    
    num_classes = dataloaders["num_classes"]
    
    # Create manager
    manager = ChampionChallengerManager(
        config=config,
        promotion_threshold=promotion_threshold,
    )
    
    # Train challenger
    challenger_path, challenger_metrics = manager.train_challenger(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        num_classes=num_classes,
    )
    
    logger.info(f"Challenger trained: {challenger_path}")
    logger.info(f"Challenger F1: {challenger_metrics.f1_macro:.4f}")
    
    # Evaluate on test set for fair comparison
    challenger_test_metrics = manager.evaluate_model(
        challenger_path,
        dataloaders["test"],
        num_classes,
    )
    
    # Compare and potentially promote
    if force_promote:
        logger.info("Force promoting challenger...")
        manager._promote_challenger()
        result = RetrainingResult(
            champion_metrics={},
            challenger_metrics=challenger_test_metrics.to_dict(),
            winner="challenger",
            promoted=True,
            improvement=0,
            primary_metric="f1_macro",
            threshold=promotion_threshold,
            timestamp=datetime.now().isoformat(),
            champion_path=str(manager.champion_dir / "model.pt"),
            challenger_path=challenger_path,
        )
    else:
        result = manager.compare_and_promote(
            challenger_test_metrics,
            test_loader=dataloaders["test"],
            num_classes=num_classes,
        )
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Champion/Challenger Retraining")
    parser.add_argument("--config", "-c", default="configs/config.yaml")
    parser.add_argument("--force-promote", action="store_true", 
                       help="Force promote challenger regardless of performance")
    parser.add_argument("--threshold", type=float, default=0.01,
                       help="Minimum improvement for promotion (default: 1%%)")
    parser.add_argument("--rollback", nargs="?", const="latest",
                       help="Rollback to previous champion (optionally specify archive)")
    parser.add_argument("--output", "-o", help="Output path for results JSON")
    
    args = parser.parse_args()
    
    setup_logging()
    config = Config.from_yaml(args.config)
    
    # Handle rollback
    if args.rollback:
        manager = ChampionChallengerManager(config)
        archive = None if args.rollback == "latest" else args.rollback
        success = manager.rollback(archive)
        return 0 if success else 1
    
    # Run retraining
    result = retrain_pipeline(
        config=config,
        force_promote=args.force_promote,
        promotion_threshold=args.threshold,
    )
    
    # Print results
    print("\n" + "="*60)
    print("RETRAINING RESULTS")
    print("="*60)
    print(f"Winner: {result.winner.upper()}")
    print(f"Promoted: {'Yes' if result.promoted else 'No'}")
    print(f"Improvement: {result.improvement:+.2f}%")
    print(f"Threshold: {result.threshold*100:.1f}%")
    print(f"\nChampion path: {result.champion_path}")
    
    if result.promoted:
        print("\n✓ New champion model deployed!")
    else:
        print("\n✗ Champion model retained (challenger did not meet threshold)")
    
    # Save results
    if args.output:
        result.to_json(args.output)
        print(f"\nResults saved to: {args.output}")
    
    # Log to MLflow
    if MLFLOW_AVAILABLE and config.mlflow.enabled:
        with mlflow.start_run(run_name="retraining"):
            mlflow.log_param("promoted", result.promoted)
            mlflow.log_param("winner", result.winner)
            mlflow.log_metric("improvement", result.improvement)


if __name__ == "__main__":
    main()
