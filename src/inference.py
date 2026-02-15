"""
Inference pipeline for multimodal biometric recognition.

This module provides:
- Batch inference on new data
- Single sample prediction
- Embedding extraction for retrieval
- Output to various formats (CSV, Parquet, Delta)
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .utils import Config, setup_logging, get_device, ensure_dir
from .data_loader import BiometricDataset
from .model import MultimodalBiometricModel
from .model_registry import get_model_registry, UnityCatalogRegistry

logger = logging.getLogger("biometric_mlops")


class InferencePipeline:
    """
    Inference pipeline for biometric recognition.
    
    Handles:
    - Loading trained model from checkpoint or registry
    - Batch and single-sample inference
    - Embedding extraction
    - Result export
    
    Args:
        checkpoint_path: Path to model checkpoint (use None for registry)
        device: Inference device
        config: Optional config for additional settings
        registry_alias: Load from registry with this alias
        registry_version: Load specific version from registry
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        config: Optional[Config] = None,
        registry_alias: Optional[str] = None,
        registry_version: Optional[int] = None,
    ):
        self.device = device or get_device("auto")
        self.config = config
        
        # Determine source and load model
        if registry_alias or registry_version:
            # Load from registry
            self.model = self._load_from_registry(registry_alias, registry_version)
            self.checkpoint = None
        elif checkpoint_path:
            # Load from checkpoint file
            self.checkpoint = self._load_checkpoint(checkpoint_path)
            self.model = self._build_model()
        else:
            # Default: try registry champion, then fall back to checkpoint
            self.model = self._load_default()
            self.checkpoint = None
        
        # Store metadata
        self.num_classes = self.model.num_classes
        
        logger.info(f"Inference pipeline ready on {self.device}")
    
    def _load_from_registry(
        self,
        alias: Optional[str] = None,
        version: Optional[int] = None,
    ) -> MultimodalBiometricModel:
        """Load model from Unity Catalog registry."""
        if self.config is None:
            raise ValueError("Config required for registry loading")
        
        registry = get_model_registry(self.config)
        
        if version:
            model = registry.load_model(version=version)
            logger.info(f"Loaded model from registry: version {version}")
        else:
            alias = alias or "champion"
            model = registry.load_model(alias=alias)
            logger.info(f"Loaded model from registry: alias '{alias}'")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_default(self) -> MultimodalBiometricModel:
        """Load model from default location (registry first, then checkpoint)."""
        # Try registry first if config available
        if self.config:
            registry_config = getattr(self.config, "model_registry", None)
            if registry_config and getattr(registry_config, "enabled", False):
                try:
                    return self._load_from_registry(alias="champion")
                except Exception as e:
                    logger.warning(f"Could not load from registry: {e}")
        
        # Fall back to checkpoint
        checkpoint_path = Path("outputs/checkpoints/best_model.pt")
        if checkpoint_path.exists():
            self.checkpoint = self._load_checkpoint(checkpoint_path)
            return self._build_model()
        
        raise FileNotFoundError(
            "No model found. Provide checkpoint_path, registry_alias, or train a model first."
        )
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Handle "latest" keyword
        if str(checkpoint_path) == "latest":
            checkpoint_dir = Path("outputs/checkpoints")
            checkpoint_path = checkpoint_dir / "best_model.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint
    
    def _build_model(self) -> MultimodalBiometricModel:
        """Build and load model from checkpoint."""
        # Extract model config from checkpoint
        ckpt_config = self.checkpoint.get("config", {})
        model_config = ckpt_config.get("model", {})
        
        # Get number of classes from checkpoint
        state_dict = self.checkpoint["model_state_dict"]
        # Infer num_classes from classifier weight shape
        classifier_weight_key = "classifier.3.weight"
        if classifier_weight_key in state_dict:
            num_classes = state_dict[classifier_weight_key].shape[0]
        else:
            num_classes = model_config.get("num_classes", 100)
        
        model = MultimodalBiometricModel(
            num_classes=num_classes,
            embedding_dim=model_config.get("embedding_dim", 128),
            dropout=model_config.get("dropout", 0.3),
            fusion_method=model_config.get("fusion_method", "concat")
        )
        
        model.load_state_dict(self.checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
        return_probs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on input tensors.
        
        Args:
            iris: Iris images [B, 1, H, W]
            fingerprint: Fingerprint images [B, 1, H, W]
            return_probs: Whether to return probabilities
            
        Returns:
            Dict with predictions and optional probabilities
        """
        iris = iris.to(self.device)
        fingerprint = fingerprint.to(self.device)
        
        output = self.model(iris, fingerprint)
        logits = output["logits"]
        
        predictions = logits.argmax(dim=1)
        
        result = {"predictions": predictions.cpu()}
        
        if return_probs:
            probs = F.softmax(logits, dim=1)
            result["probabilities"] = probs.cpu()
            result["confidence"] = probs.max(dim=1)[0].cpu()
        
        return result
    
    @torch.no_grad()
    def get_embeddings(
        self,
        iris: torch.Tensor,
        fingerprint: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract fused embeddings for retrieval/matching.
        
        Args:
            iris: Iris images [B, 1, H, W]
            fingerprint: Fingerprint images [B, 1, H, W]
            
        Returns:
            Fused embeddings [B, embedding_dim]
        """
        iris = iris.to(self.device)
        fingerprint = fingerprint.to(self.device)
        
        embeddings = self.model.get_embeddings(iris, fingerprint)
        return embeddings.cpu()
    
    def predict_dataloader(
        self,
        dataloader: DataLoader,
        return_embeddings: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on entire dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dict with all predictions and metadata
        """
        all_predictions = []
        all_labels = []
        all_subject_ids = []
        all_confidences = []
        all_embeddings = []
        
        for batch in dataloader:
            result = self.predict(
                batch["iris"],
                batch["fingerprint"],
                return_probs=True
            )
            
            all_predictions.append(result["predictions"].numpy())
            all_confidences.append(result["confidence"].numpy())
            all_labels.append(batch["label"].numpy())
            all_subject_ids.extend(batch["subject_id"])
            
            if return_embeddings:
                embeddings = self.get_embeddings(
                    batch["iris"],
                    batch["fingerprint"]
                )
                all_embeddings.append(embeddings.numpy())
        
        results = {
            "predictions": np.concatenate(all_predictions),
            "labels": np.concatenate(all_labels),
            "confidences": np.concatenate(all_confidences),
            "subject_ids": np.array(all_subject_ids)
        }
        
        if return_embeddings:
            results["embeddings"] = np.concatenate(all_embeddings)
        
        # Calculate accuracy
        results["accuracy"] = (results["predictions"] == results["labels"]).mean()
        
        return results
    
    def predict_dataset(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 64,
        num_workers: int = 4,
        return_embeddings: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a data directory.
        
        Args:
            data_dir: Path to data directory
            batch_size: Inference batch size
            num_workers: DataLoader workers
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dict with predictions
        """
        dataset = BiometricDataset(data_dir)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return self.predict_dataloader(dataloader, return_embeddings)
    
    def export_predictions(
        self,
        results: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        format: str = "parquet"
    ) -> None:
        """
        Export predictions to file.
        
        Args:
            results: Prediction results dict
            output_path: Output file path
            format: Output format ("parquet", "csv")
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        # Build dataframe
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for export")
        
        df = pd.DataFrame({
            "subject_id": results["subject_ids"],
            "prediction": results["predictions"],
            "label": results["labels"],
            "confidence": results["confidences"],
            "correct": results["predictions"] == results["labels"]
        })
        
        if format == "parquet":
            if not PYARROW_AVAILABLE:
                raise ImportError("pyarrow required for parquet export")
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Exported {len(df)} predictions to {output_path}")


def run_inference(
    config_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    registry_alias: Optional[str] = None,
    registry_version: Optional[int] = None,
) -> Dict:
    """
    Run inference from config and checkpoint or registry.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint (optional if using registry)
        data_dir: Optional data directory (defaults to config)
        output_path: Optional output path (defaults to config)
        registry_alias: Load model from registry by alias
        registry_version: Load specific version from registry
        
    Returns:
        Prediction results dict
    """
    config = Config.from_yaml(config_path)
    
    # Setup logging
    setup_logging(level=config.logging.level)
    
    # Create pipeline
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        config=config,
        registry_alias=registry_alias,
        registry_version=registry_version,
    )
    
    # Get data directory
    data_dir = data_dir or config.data.raw_dir
    
    # Run inference
    results = pipeline.predict_dataset(
        data_dir,
        batch_size=config.inference.batch_size,
        num_workers=config.runtime.num_workers
    )
    
    logger.info(f"Inference accuracy: {results['accuracy']:.4f}")
    
    # Export if output path provided
    output_path = output_path or config.inference.output_path
    if output_path:
        output_format = config.inference.output_format
        pipeline.export_predictions(results, output_path, output_format)
    
    return results


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="Run biometric model inference")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (optional if using registry)"
    )
    parser.add_argument(
        "--alias", type=str, choices=["champion", "challenger"],
        help="Load model from registry by alias"
    )
    parser.add_argument(
        "--version", type=int, default=None,
        help="Load specific version from registry"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory for inference"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for predictions"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.checkpoint and not args.alias and not args.version:
        print("No model source specified. Will try registry champion, then checkpoint.")
    
    results = run_inference(
        args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        registry_alias=args.alias,
        registry_version=args.version,
    )
    
    print("\n" + "="*50)
    print("INFERENCE COMPLETE")
    print("="*50)
    print(f"Total samples: {len(results['predictions'])}")
    print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
