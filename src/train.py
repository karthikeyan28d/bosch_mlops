"""
Training pipeline for multimodal biometric recognition.

This module provides:
- Complete training loop with validation
- Checkpointing and early stopping
- MLflow experiment tracking
- Reproducibility support
"""

import os
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .utils import (
    Config, setup_logging, set_seed, get_device, 
    ensure_dir, AverageMeter, is_databricks
)
from .data_loader import create_dataloaders
from .model import MultimodalBiometricModel, create_model_from_config
from .model_registry import get_model_registry, UnityCatalogRegistry

logger = logging.getLogger("biometric_mlops")


class Trainer:
    """
    Training manager for multimodal biometric model.
    
    Handles:
    - Training and validation loops
    - Checkpointing
    - Early stopping
    - MLflow logging
    - Mixed precision training
    
    Args:
        model: PyTorch model
        config: Configuration object
        device: Training device
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training config
        train_config = config.training
        self.epochs = train_config.epochs
        self.lr = train_config.learning_rate
        self.grad_clip = train_config.gradient_clip
        self.mixed_precision = train_config.mixed_precision
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(train_config)
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler(train_config)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Checkpointing
        self.checkpoint_dir = ensure_dir(train_config.checkpoint.save_dir)
        self.save_best_only = train_config.checkpoint.save_best_only
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_enabled = train_config.early_stopping.enabled
        self.patience = train_config.early_stopping.patience
        self.min_delta = train_config.early_stopping.min_delta
        self.patience_counter = 0
        
        # MLflow
        self.mlflow_enabled = config.mlflow.enabled and MLFLOW_AVAILABLE
        if self.mlflow_enabled:
            self._setup_mlflow(config.mlflow)
    
    def _setup_optimizer(self, train_config) -> optim.Optimizer:
        """Setup optimizer based on config."""
        optimizer_name = train_config.optimizer.lower()
        
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=train_config.weight_decay
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=train_config.weight_decay
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=train_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self, train_config) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = train_config.scheduler.lower()
        params = train_config.scheduler_params.to_dict() if hasattr(
            train_config.scheduler_params, 'to_dict'
        ) else dict(train_config.scheduler_params._data)
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get("T_max", self.epochs),
                eta_min=params.get("eta_min", 1e-6)
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 10),
                gamma=params.get("gamma", 0.1)
            )
        elif scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _setup_mlflow(self, mlflow_config) -> None:
        """Setup MLflow tracking."""
        if mlflow_config.tracking_uri:
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        
        mlflow.set_experiment(mlflow_config.experiment_name)
        
        # Start run
        run_name = f"{mlflow_config.run_name_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        # Log config
        mlflow.log_params({
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "optimizer": self.config.training.optimizer,
            "batch_size": self.config.training.batch_size,
            "model_embedding_dim": self.config.model.embedding_dim,
            "fusion_method": self.config.model.fusion_method,
        })
        
        logger.info(f"MLflow run started: {run_name}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run a single training epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Accuracy")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            iris = batch["iris"].to(self.device)
            fingerprint = batch["fingerprint"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    output = self.model(iris, fingerprint)
                    loss = self.criterion(output["logits"], labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(iris, fingerprint)
                loss = self.criterion(output["logits"], labels)
                
                # Backward pass
                loss.backward()
                
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # Calculate accuracy
            preds = output["logits"].argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            
            # Update meters
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc, labels.size(0))
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg:.4f}")
        
        return {
            "train_loss": loss_meter.avg,
            "train_acc": acc_meter.avg
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Accuracy")
        
        for batch in val_loader:
            iris = batch["iris"].to(self.device)
            fingerprint = batch["fingerprint"].to(self.device)
            labels = batch["label"].to(self.device)
            
            output = self.model(iris, fingerprint)
            loss = self.criterion(output["logits"], labels)
            
            preds = output["logits"].argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc, labels.size(0))
        
        return {
            "val_loss": loss_meter.avg,
            "val_acc": acc_meter.avg
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict()
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest
        if not self.save_best_only:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model (epoch {epoch})")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict:
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def _register_model_to_catalog(self, metrics: Dict[str, float]) -> None:
        """
        Register trained model to Unity Catalog.
        
        Args:
            metrics: Training metrics for the model
        """
        # Check if model registry is enabled
        registry_config = getattr(self.config, "model_registry", None)
        if not registry_config or not getattr(registry_config, "enabled", False):
            logger.info("Model registry disabled, skipping registration")
            return
        
        try:
            registry = get_model_registry(self.config)
            
            # Determine alias (new models register as challenger by default)
            default_alias = getattr(registry_config, "default_alias", "challenger")
            
            # Register the model
            result = registry.register_model(
                model=self.model,
                metrics=metrics,
                alias=default_alias
            )
            
            logger.info(
                f"Model registered to Unity Catalog: {result.model_name} "
                f"version {result.version} with alias '{result.alias}'"
            )
            
            # Auto-promote if configured and metrics exceed threshold
            auto_promote = getattr(registry_config, "auto_register", False)
            if auto_promote and default_alias == "challenger":
                should_promote, details = registry.should_promote(
                    threshold=getattr(registry_config, "promotion_threshold", 0.01)
                )
                
                if should_promote:
                    registry.promote_challenger()
                    logger.info(
                        f"Model auto-promoted to champion: "
                        f"{details.get('improvement', 0):.4f} improvement"
                    )
                else:
                    logger.info(
                        f"Model not promoted: {details.get('reason', 'no improvement')}"
                    )
            
            # Log to MLflow
            if self.mlflow_enabled:
                mlflow.log_params({
                    "registered_model_name": result.model_name,
                    "registered_version": result.version,
                    "registered_alias": result.alias
                })
        
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            # Don't fail training if registration fails
            if is_databricks():
                raise  # Re-raise in Databricks for visibility
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Run full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dict with final metrics
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        start_time = time.time()
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            for key in history:
                if key in metrics:
                    history[key].append(metrics[key])
            
            # Check for best model
            is_best = val_metrics["val_loss"] < self.best_val_loss - self.min_delta
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_acc']:.4f}"
            )
            
            if self.mlflow_enabled:
                mlflow.log_metrics(metrics, step=epoch)
            
            # Early stopping check
            if self.early_stopping_enabled and self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(f"Best model at epoch {self.best_epoch} with val_loss={self.best_val_loss:.4f}")
        
        # Log final artifacts
        if self.mlflow_enabled:
            mlflow.log_artifact(str(self.checkpoint_dir / "best_model.pt"))
        
        # Final metrics for registry
        final_metrics = {
            "val_loss": self.best_val_loss,
            "val_acc": history["val_acc"][self.best_epoch - 1] if history["val_acc"] else 0.0,
            "train_loss": history["train_loss"][self.best_epoch - 1] if history["train_loss"] else 0.0,
            "best_epoch": self.best_epoch,
            "training_time": total_time
        }
        
        # Register model in Unity Catalog
        self._register_model_to_catalog(final_metrics)
        
        if self.mlflow_enabled:
            mlflow.end_run()
        
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "training_time": total_time,
            "history": history
        }


def train_from_config(config_path: Union[str, Path], resume: Optional[str] = None) -> Dict:
    """
    Run training from configuration file.
    
    Args:
        config_path: Path to config YAML
        resume: Optional checkpoint path to resume from
        
    Returns:
        Training results dict
    """
    # Load config
    config = Config.from_yaml(config_path)
    
    # Setup logging
    log_file = config.logging.file_path if config.logging.file_path else None
    setup_logging(level=config.logging.level, log_file=log_file)
    
    # Set seed for reproducibility
    set_seed(config.runtime.seed)
    
    # Get device
    device = get_device(config.runtime.device)
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        data_dir=config.data.raw_dir,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers,
        split_ratios=tuple(config.data.split.to_dict().values()) if hasattr(config.data.split, 'to_dict') else (0.7, 0.15, 0.15),
        image_size=tuple(config.data.image.size),
        seed=config.runtime.seed
    )
    
    logger.info(f"Data loaded: {num_classes} classes")
    
    # Create model
    model = create_model_from_config(config, num_classes)
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Resume from checkpoint if provided
    if resume:
        trainer.load_checkpoint(resume)
    
    # Train
    results = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test metrics: Loss={test_metrics['val_loss']:.4f}, "
               f"Acc={test_metrics['val_acc']:.4f}")
    
    results["test_loss"] = test_metrics["val_loss"]
    results["test_acc"] = test_metrics["val_acc"]
    
    return results


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train biometric recognition model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    results = train_from_config(args.config, args.resume)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Test accuracy: {results.get('test_acc', 'N/A'):.4f}")
    print(f"Training time: {results['training_time']:.1f}s")


if __name__ == "__main__":
    main()
