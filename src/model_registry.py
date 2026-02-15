"""
Model Registry Module with Unity Catalog Integration.

Manages model versioning, aliases, and lifecycle using:
- Unity Catalog (Databricks) - Production
- MLflow Model Registry - Local/Staging

Key concepts:
- Champion: Production model (alias: "champion")
- Challenger: Staged model (alias: "challenger")  
- Tags: champion, challenger, archived, version metadata
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

import torch
import torch.nn as nn

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MlflowClient = None
    MlflowException = Exception

logger = logging.getLogger("biometric_mlops")


# =============================================================================
# Constants & Enums
# =============================================================================

class ModelAlias(str, Enum):
    """Model aliases for lifecycle management."""
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    ARCHIVED = "archived"


class ModelStage(str, Enum):
    """MLflow model stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class RegistryType(str, Enum):
    """Type of model registry."""
    UNITY_CATALOG = "unity_catalog"
    MLFLOW = "mlflow"
    LOCAL = "local"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelVersion:
    """Represents a registered model version."""
    version: int
    name: str
    run_id: Optional[str] = None
    stage: str = "None"
    alias: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: str = ""
    description: Optional[str] = None
    source: Optional[str] = None
    
    @classmethod
    def from_mlflow(cls, mv, client: Optional[Any] = None) -> "ModelVersion":
        """Create from MLflow ModelVersion object."""
        tags = dict(mv.tags) if mv.tags else {}
        alias = tags.get("alias", tags.get("model_type"))
        
        # Load metrics from run
        metrics = {}
        if client and mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                metrics = dict(run.data.metrics)
            except Exception:
                pass
        
        return cls(
            version=int(mv.version),
            name=mv.name,
            run_id=mv.run_id,
            stage=mv.current_stage or "None",
            alias=alias,
            tags=tags,
            metrics=metrics,
            created_at=str(mv.creation_timestamp),
            description=mv.description,
            source=mv.source,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def is_champion(self) -> bool:
        return self.alias == ModelAlias.CHAMPION.value
    
    @property
    def is_challenger(self) -> bool:
        return self.alias == ModelAlias.CHALLENGER.value


@dataclass
class RegistrationResult:
    """Result of model registration."""
    success: bool
    version: Optional[int] = None
    model_uri: Optional[str] = None
    alias: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Unity Catalog Model Registry
# =============================================================================

class UnityCatalogRegistry:
    """
    Unity Catalog Model Registry for Databricks.
    
    Model naming: <catalog>.<schema>.<model_name>
    Example: ml_prod.biometrics.multimodal_model
    
    Features:
    - Register models with champion/challenger aliases
    - Load models by alias or version
    - Promote/demote models between aliases
    - Full MLflow compatibility
    """
    
    def __init__(
        self,
        catalog: str,
        schema: str,
        model_name: str,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize Unity Catalog registry.
        
        Args:
            catalog: Unity Catalog name (e.g., "ml_prod")
            schema: Schema name (e.g., "biometrics")
            model_name: Model name (e.g., "multimodal_model")
            tracking_uri: MLflow tracking URI (auto-detected on Databricks)
        """
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow required. Install: pip install mlflow")
        
        self.catalog = catalog
        self.schema = schema
        self.model_name = model_name
        self.full_name = f"{catalog}.{schema}.{model_name}"
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif self._is_databricks():
            mlflow.set_tracking_uri("databricks")
        
        # Set registry URI for Unity Catalog
        if self._is_databricks():
            mlflow.set_registry_uri("databricks-uc")
        
        self.client = MlflowClient()
        
        # Ensure model is registered
        self._ensure_registered()
    
    def _is_databricks(self) -> bool:
        """Check if running on Databricks."""
        return (
            "DATABRICKS_RUNTIME_VERSION" in os.environ or
            "SPARK_HOME" in os.environ and "databricks" in os.environ.get("SPARK_HOME", "").lower()
        )
    
    def _ensure_registered(self) -> None:
        """Ensure model exists in registry."""
        try:
            self.client.get_registered_model(self.full_name)
            logger.debug(f"Model exists: {self.full_name}")
        except MlflowException:
            try:
                self.client.create_registered_model(
                    name=self.full_name,
                    description=f"Multimodal biometric model - {self.model_name}",
                    tags={"created_by": "biometric_mlops"}
                )
                logger.info(f"Created registered model: {self.full_name}")
            except MlflowException as e:
                logger.warning(f"Could not create model: {e}")
    
    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------
    
    def register_model(
        self,
        model: nn.Module,
        metrics: Dict[str, float],
        alias: str = ModelAlias.CHALLENGER.value,
        tags: Optional[Dict[str, str]] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> RegistrationResult:
        """
        Register a PyTorch model to Unity Catalog.
        
        Args:
            model: PyTorch model to register
            metrics: Model metrics (accuracy, f1, etc.)
            alias: Model alias (champion/challenger)
            tags: Additional tags
            signature: MLflow model signature
            input_example: Example input for signature inference
            run_id: Existing MLflow run ID (creates new if None)
            experiment_name: MLflow experiment name
            
        Returns:
            RegistrationResult with version info
        """
        try:
            all_tags = {
                "alias": alias,
                "registered_at": datetime.now().isoformat(),
                "framework": "pytorch",
            }
            if tags:
                all_tags.update(tags)
            
            # Add metrics to tags for quick reference
            for key, value in metrics.items():
                all_tags[f"metric_{key}"] = str(round(value, 4))
            
            # Set experiment
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            # Start or use existing run
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    return self._register_in_run(model, metrics, alias, all_tags, signature, input_example)
            else:
                with mlflow.start_run(run_name=f"register_{alias}"):
                    return self._register_in_run(model, metrics, alias, all_tags, signature, input_example)
                    
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return RegistrationResult(
                success=False,
                error=str(e)
            )
    
    def _register_in_run(
        self,
        model: nn.Module,
        metrics: Dict[str, float],
        alias: str,
        tags: Dict[str, str],
        signature: Optional[Any],
        input_example: Optional[Any],
    ) -> RegistrationResult:
        """Register model within MLflow run context."""
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log parameters
        mlflow.log_params({
            "model_alias": alias,
            "model_name": self.full_name,
        })
        
        # Log model
        model_info = mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=self.full_name,
        )
        
        # Get version number
        run_id = mlflow.active_run().info.run_id
        versions = self.client.search_model_versions(f"name='{self.full_name}' AND run_id='{run_id}'")
        
        if not versions:
            return RegistrationResult(success=False, error="Version not found after registration")
        
        version = int(versions[0].version)
        
        # Set tags on version
        for key, value in tags.items():
            self.client.set_model_version_tag(self.full_name, version, key, value)
        
        # Set alias (MLflow 2.x / Unity Catalog)
        try:
            # Remove existing alias from other versions
            self._clear_alias(alias)
            # Set alias on new version
            self.client.set_registered_model_alias(self.full_name, alias, version)
            logger.info(f"Set alias '{alias}' on version {version}")
        except AttributeError:
            # Fallback: use tags only
            logger.warning("Aliases not supported, using tags only")
        
        model_uri = f"models:/{self.full_name}@{alias}"
        
        logger.info(f"Registered model: {self.full_name} v{version} as {alias}")
        
        return RegistrationResult(
            success=True,
            version=version,
            model_uri=model_uri,
            alias=alias,
        )
    
    def register_from_checkpoint(
        self,
        checkpoint_path: str,
        model_class: type,
        model_kwargs: Dict[str, Any],
        metrics: Dict[str, float],
        alias: str = ModelAlias.CHALLENGER.value,
        tags: Optional[Dict[str, str]] = None,
    ) -> RegistrationResult:
        """
        Register model from checkpoint file.
        
        Args:
            checkpoint_path: Path to .pt checkpoint
            model_class: Model class to instantiate
            model_kwargs: Arguments for model constructor
            metrics: Model metrics
            alias: Model alias
            tags: Additional tags
            
        Returns:
            RegistrationResult
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Create model
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Add checkpoint info to tags
        all_tags = tags or {}
        all_tags["checkpoint_path"] = checkpoint_path
        all_tags["checkpoint_epoch"] = str(checkpoint.get("epoch", "unknown"))
        
        return self.register_model(model, metrics, alias, all_tags)
    
    def _clear_alias(self, alias: str) -> None:
        """Remove alias from all versions."""
        try:
            # Try to get current version with alias
            self.client.delete_registered_model_alias(self.full_name, alias)
        except (MlflowException, AttributeError):
            pass  # Alias doesn't exist or not supported
    
    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------
    
    def load_model(
        self,
        alias: Optional[str] = None,
        version: Optional[int] = None,
    ) -> nn.Module:
        """
        Load model from Unity Catalog.
        
        Args:
            alias: Model alias (champion/challenger)
            version: Specific version number
            
        Returns:
            Loaded PyTorch model
        """
        if alias:
            model_uri = f"models:/{self.full_name}@{alias}"
        elif version:
            model_uri = f"models:/{self.full_name}/{version}"
        else:
            # Default to champion, fallback to latest
            try:
                model_uri = f"models:/{self.full_name}@{ModelAlias.CHAMPION.value}"
                return mlflow.pytorch.load_model(model_uri)
            except MlflowException:
                model_uri = f"models:/{self.full_name}/latest"
        
        logger.info(f"Loading model: {model_uri}")
        return mlflow.pytorch.load_model(model_uri)
    
    def load_champion(self) -> nn.Module:
        """Load champion (production) model."""
        return self.load_model(alias=ModelAlias.CHAMPION.value)
    
    def load_challenger(self) -> nn.Module:
        """Load challenger (staging) model."""
        return self.load_model(alias=ModelAlias.CHALLENGER.value)
    
    # -------------------------------------------------------------------------
    # Version Management
    # -------------------------------------------------------------------------
    
    def get_version(
        self,
        alias: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Optional[ModelVersion]:
        """Get model version info."""
        try:
            if alias:
                mv = self.client.get_model_version_by_alias(self.full_name, alias)
                return ModelVersion.from_mlflow(mv, self.client)
            elif version:
                mv = self.client.get_model_version(self.full_name, str(version))
                return ModelVersion.from_mlflow(mv, self.client)
        except (MlflowException, AttributeError):
            # Fallback: search by tags
            return self._get_version_by_tag(alias or str(version))
        return None
    
    def _get_version_by_tag(self, tag_value: str) -> Optional[ModelVersion]:
        """Find version by tag (fallback method)."""
        try:
            versions = self.client.search_model_versions(f"name='{self.full_name}'")
            for mv in versions:
                if mv.tags and mv.tags.get("alias") == tag_value:
                    return ModelVersion.from_mlflow(mv, self.client)
        except MlflowException:
            pass
        return None
    
    def get_champion(self) -> Optional[ModelVersion]:
        """Get champion model version."""
        return self.get_version(alias=ModelAlias.CHAMPION.value)
    
    def get_challenger(self) -> Optional[ModelVersion]:
        """Get challenger model version."""
        return self.get_version(alias=ModelAlias.CHALLENGER.value)
    
    def list_versions(
        self,
        include_archived: bool = False,
        max_results: int = 100,
    ) -> List[ModelVersion]:
        """List all model versions."""
        try:
            versions = self.client.search_model_versions(
                f"name='{self.full_name}'",
                max_results=max_results,
            )
            result = []
            for mv in versions:
                model_version = ModelVersion.from_mlflow(mv, self.client)
                if include_archived or not model_version.is_archived:
                    result.append(model_version)
            return sorted(result, key=lambda x: x.version, reverse=True)
        except MlflowException:
            return []
    
    # -------------------------------------------------------------------------
    # Promotion & Lifecycle
    # -------------------------------------------------------------------------
    
    def promote_challenger(self) -> bool:
        """
        Promote challenger to champion.
        
        - Archives current champion
        - Sets challenger as new champion
        
        Returns:
            True if promotion successful
        """
        challenger = self.get_challenger()
        if not challenger:
            logger.error("No challenger model found to promote")
            return False
        
        champion = self.get_champion()
        
        # Archive current champion
        if champion:
            self._archive_version(champion.version)
        
        # Promote challenger
        try:
            # Update alias
            self._clear_alias(ModelAlias.CHAMPION.value)
            self.client.set_registered_model_alias(
                self.full_name, 
                ModelAlias.CHAMPION.value, 
                challenger.version
            )
            
            # Update tags
            self.client.set_model_version_tag(
                self.full_name, challenger.version, 
                "alias", ModelAlias.CHAMPION.value
            )
            self.client.set_model_version_tag(
                self.full_name, challenger.version,
                "promoted_at", datetime.now().isoformat()
            )
            self.client.set_model_version_tag(
                self.full_name, challenger.version,
                "previous_alias", ModelAlias.CHALLENGER.value
            )
            
            # Remove challenger alias
            self._clear_alias(ModelAlias.CHALLENGER.value)
            
            logger.info(f"Promoted version {challenger.version} to champion")
            return True
            
        except MlflowException as e:
            logger.error(f"Promotion failed: {e}")
            return False
    
    def _archive_version(self, version: int) -> None:
        """Archive a model version."""
        archive_alias = f"{ModelAlias.ARCHIVED.value}_{version}"
        
        try:
            self.client.set_registered_model_alias(
                self.full_name, archive_alias, version
            )
        except (MlflowException, AttributeError):
            pass
        
        # Update tags
        self.client.set_model_version_tag(
            self.full_name, version,
            "alias", ModelAlias.ARCHIVED.value
        )
        self.client.set_model_version_tag(
            self.full_name, version,
            "archived_at", datetime.now().isoformat()
        )
        
        logger.info(f"Archived version {version}")
    
    def rollback(self, to_version: Optional[int] = None) -> bool:
        """
        Rollback champion to previous version.
        
        Args:
            to_version: Specific version (default: previous champion)
            
        Returns:
            True if rollback successful
        """
        if to_version:
            target = to_version
        else:
            # Find most recent archived version
            versions = self.list_versions(include_archived=True)
            archived = [v for v in versions if v.is_archived]
            if not archived:
                logger.error("No archived versions for rollback")
                return False
            target = archived[0].version
        
        try:
            # Archive current champion
            champion = self.get_champion()
            if champion:
                self._archive_version(champion.version)
            
            # Set target as champion
            self._clear_alias(ModelAlias.CHAMPION.value)
            self.client.set_registered_model_alias(
                self.full_name,
                ModelAlias.CHAMPION.value,
                target
            )
            self.client.set_model_version_tag(
                self.full_name, target,
                "alias", ModelAlias.CHAMPION.value
            )
            self.client.set_model_version_tag(
                self.full_name, target,
                "rolled_back_at", datetime.now().isoformat()
            )
            
            logger.info(f"Rolled back to version {target}")
            return True
            
        except MlflowException as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    
    def compare_versions(
        self,
        version_a: int,
        version_b: int,
        primary_metric: str = "f1_macro",
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Returns dict with comparison results.
        """
        mv_a = self.get_version(version=version_a)
        mv_b = self.get_version(version=version_b)
        
        if not mv_a or not mv_b:
            return {"error": "One or both versions not found"}
        
        metric_a = mv_a.metrics.get(primary_metric, 0)
        metric_b = mv_b.metrics.get(primary_metric, 0)
        
        improvement = (metric_b - metric_a) / metric_a if metric_a > 0 else 0
        
        return {
            "version_a": version_a,
            "version_b": version_b,
            "metric": primary_metric,
            "value_a": metric_a,
            "value_b": metric_b,
            "improvement": improvement,
            "improvement_pct": improvement * 100,
            "winner": "B" if metric_b > metric_a else "A",
        }
    
    def should_promote(
        self,
        threshold: float = 0.01,
        primary_metric: str = "f1_macro",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if challenger should be promoted.
        
        Args:
            threshold: Minimum improvement required (0.01 = 1%)
            primary_metric: Metric to compare
            
        Returns:
            Tuple of (should_promote, comparison_details)
        """
        champion = self.get_champion()
        challenger = self.get_challenger()
        
        if not challenger:
            return False, {"error": "No challenger found"}
        
        if not champion:
            # No champion - auto promote
            return True, {"reason": "No existing champion"}
        
        comparison = self.compare_versions(
            champion.version,
            challenger.version,
            primary_metric
        )
        
        should_promote = comparison["improvement"] > threshold
        comparison["threshold"] = threshold
        comparison["should_promote"] = should_promote
        
        return should_promote, comparison


# =============================================================================
# Factory Function
# =============================================================================

def get_model_registry(
    config: Any,
    registry_type: Optional[str] = None,
) -> UnityCatalogRegistry:
    """
    Factory function to get appropriate model registry.
    
    Args:
        config: Configuration object
        registry_type: Override registry type
        
    Returns:
        Model registry instance
    """
    # Get registry config
    reg_config = getattr(config, "model_registry", None) or {}
    if hasattr(reg_config, "__dict__"):
        reg_config = reg_config.__dict__
    
    catalog = reg_config.get("catalog", "ml_models")
    schema = reg_config.get("schema", "biometrics")
    model_name = reg_config.get("model_name", "multimodal_biometric")
    tracking_uri = reg_config.get("tracking_uri")
    
    # Check environment
    is_databricks = (
        "DATABRICKS_RUNTIME_VERSION" in os.environ or
        registry_type == RegistryType.UNITY_CATALOG.value
    )
    
    if is_databricks:
        return UnityCatalogRegistry(
            catalog=catalog,
            schema=schema,
            model_name=model_name,
            tracking_uri=tracking_uri,
        )
    else:
        # Local MLflow - use simple name
        return UnityCatalogRegistry(
            catalog="local",
            schema="dev",
            model_name=model_name,
            tracking_uri=tracking_uri or "mlruns",
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for model registry operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--catalog", default="ml_models")
    parser.add_argument("--schema", default="biometrics")
    parser.add_argument("--model", default="multimodal_biometric")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # List versions
    list_parser = subparsers.add_parser("list", help="List model versions")
    list_parser.add_argument("--all", action="store_true", help="Include archived")
    
    # Get info
    info_parser = subparsers.add_parser("info", help="Get version info")
    info_parser.add_argument("--alias", help="Model alias")
    info_parser.add_argument("--version", type=int, help="Version number")
    
    # Promote
    promote_parser = subparsers.add_parser("promote", help="Promote challenger")
    
    # Rollback
    rollback_parser = subparsers.add_parser("rollback", help="Rollback champion")
    rollback_parser.add_argument("--version", type=int, help="Target version")
    
    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare versions")
    compare_parser.add_argument("--threshold", type=float, default=0.01)
    
    args = parser.parse_args()
    
    if not MLFLOW_AVAILABLE:
        print("MLflow not installed. Run: pip install mlflow")
        return 1
    
    registry = UnityCatalogRegistry(args.catalog, args.schema, args.model)
    
    if args.command == "list":
        versions = registry.list_versions(include_archived=args.all)
        print(f"\nModel: {registry.full_name}")
        print("-" * 60)
        for v in versions:
            alias_str = f"[{v.alias}]" if v.alias else ""
            print(f"  v{v.version:3d} {alias_str:12} | {v.created_at[:19]}")
    
    elif args.command == "info":
        version = registry.get_version(alias=args.alias, version=args.version)
        if version:
            print(json.dumps(version.to_dict(), indent=2))
        else:
            print("Version not found")
    
    elif args.command == "promote":
        success = registry.promote_challenger()
        print("Promoted!" if success else "Promotion failed")
    
    elif args.command == "rollback":
        success = registry.rollback(args.version)
        print("Rolled back!" if success else "Rollback failed")
    
    elif args.command == "compare":
        should_promote, details = registry.should_promote(args.threshold)
        print(json.dumps(details, indent=2))
        print(f"\nShould promote: {should_promote}")
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
