"""
Model and Data Drift Monitoring Module.

Detects data drift and model performance drift using:
- Evidently AI for comprehensive drift reports
- Statistical tests for distribution shift
- Performance degradation tracking

Key concepts:
- Data Drift: Distribution of input features has changed
- Model Drift: Model performance has degraded over time
- Concept Drift: Relationship between features and target has changed
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
    )
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
        EmbeddingsDriftMetric,
    )
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
    from evidently.tests import (
        TestColumnDrift,
        TestShareOfDriftedColumns,
        TestNumberOfDriftedColumns,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .utils import Config, setup_logging, get_device
from .data_loader import BiometricDataset
from .model import MultimodalBiometricModel
from .model_registry import get_model_registry

logger = logging.getLogger("biometric_mlops")


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_detected: bool
    drift_score: float
    drift_type: str  # "data", "model", "concept"
    metrics: Dict[str, float]
    threshold: float
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class MonitoringReport:
    """Complete monitoring report."""
    data_drift: Optional[DriftResult]
    model_drift: Optional[DriftResult]
    overall_status: str  # "healthy", "warning", "critical"
    recommendations: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "data_drift": self.data_drift.to_dict() if self.data_drift else None,
            "model_drift": self.model_drift.to_dict() if self.model_drift else None,
            "overall_status": self.overall_status,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }
        return result
    
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class FeatureExtractor:
    """Extract features from images for drift detection."""
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        config: Optional[Config] = None,
        load_from_registry: bool = False,
    ):
        self.device = device or torch.device("cpu")
        self.model = model
        self.config = config
        
        # Load model from registry if requested
        if load_from_registry and config and not model:
            self.model = self._load_from_registry()
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def _load_from_registry(self) -> Optional[torch.nn.Module]:
        """Load champion model from registry for embedding extraction."""
        try:
            registry_config = getattr(self.config, "model_registry", None)
            if registry_config and getattr(registry_config, "enabled", False):
                registry = get_model_registry(self.config)
                model = registry.load_champion()
                logger.info("Loaded champion model from registry for feature extraction")
                return model
        except Exception as e:
            logger.warning(f"Could not load model from registry: {e}")
        return None
    
    def extract_basic_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary of features
        """
        if image.ndim == 3:
            image = image.mean(axis=2)  # Convert to grayscale
        
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": float(np.min(image)),
            "max": float(np.max(image)),
            "median": float(np.median(image)),
            "q25": float(np.percentile(image, 25)),
            "q75": float(np.percentile(image, 75)),
            "skewness": float(stats.skew(image.flatten())) if SCIPY_AVAILABLE else 0,
            "kurtosis": float(stats.kurtosis(image.flatten())) if SCIPY_AVAILABLE else 0,
        }
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        iris_tensor: torch.Tensor,
        fingerprint_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Extract model embeddings for drift detection.
        
        Args:
            iris_tensor: Iris image tensor
            fingerprint_tensor: Fingerprint image tensor
            
        Returns:
            Embedding numpy array
        """
        if self.model is None:
            raise ValueError("Model required for embedding extraction")
        
        iris = iris_tensor.to(self.device)
        fingerprint = fingerprint_tensor.to(self.device)
        
        # Get embeddings (before classification layer)
        embeddings = self.model.get_embeddings(iris, fingerprint)
        return embeddings.cpu().numpy()


class DataDriftMonitor:
    """
    Monitor data drift using statistical methods and Evidently.
    
    Compares reference (training) data distribution with 
    current (production) data distribution.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_threshold: float = 0.1,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Initialize monitor.
        
        Args:
            reference_data: Reference dataset (training data features)
            drift_threshold: Threshold for drift detection (default 0.1)
            feature_columns: Columns to monitor (default: all numeric)
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for drift monitoring")
        
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.feature_columns = feature_columns or list(
            reference_data.select_dtypes(include=[np.number]).columns
        )
    
    def detect_drift_statistical(
        self,
        current_data: pd.DataFrame,
    ) -> DriftResult:
        """
        Detect drift using statistical tests (KS test).
        
        Args:
            current_data: Current data to check
            
        Returns:
            DriftResult
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for statistical drift detection")
        
        drift_scores = {}
        drifted_features = []
        
        for col in self.feature_columns:
            if col not in current_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna()
            cur_values = current_data[col].dropna()
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
            drift_scores[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drifted": p_value < self.drift_threshold,
            }
            
            if p_value < self.drift_threshold:
                drifted_features.append(col)
        
        # Calculate overall drift score
        total_features = len(drift_scores)
        drifted_count = len(drifted_features)
        drift_ratio = drifted_count / total_features if total_features > 0 else 0
        
        return DriftResult(
            drift_detected=drift_ratio > 0.2,  # >20% features drifted
            drift_score=drift_ratio,
            drift_type="data",
            metrics={
                "total_features": total_features,
                "drifted_features": drifted_count,
                "drift_ratio": drift_ratio,
            },
            threshold=self.drift_threshold,
            timestamp=datetime.now().isoformat(),
            details={
                "feature_scores": drift_scores,
                "drifted_features": drifted_features,
            },
        )
    
    def detect_drift_evidently(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> DriftResult:
        """
        Detect drift using Evidently AI.
        
        Args:
            current_data: Current data to check
            output_path: Path to save HTML report
            
        Returns:
            DriftResult
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("evidently required. Install: pip install evidently")
        
        # Create drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )
        
        # Extract results
        result_dict = report.as_dict()
        
        # Get overall dataset drift
        dataset_drift = result_dict["metrics"][0]["result"]
        drift_share = dataset_drift.get("share_of_drifted_columns", 0)
        dataset_drifted = dataset_drift.get("dataset_drift", False)
        
        # Save HTML report
        if output_path:
            report.save_html(output_path)
            logger.info(f"Drift report saved to: {output_path}")
        
        return DriftResult(
            drift_detected=dataset_drifted,
            drift_score=drift_share,
            drift_type="data",
            metrics={
                "share_of_drifted_columns": drift_share,
                "number_of_columns": dataset_drift.get("number_of_columns", 0),
                "number_of_drifted_columns": dataset_drift.get("number_of_drifted_columns", 0),
            },
            threshold=self.drift_threshold,
            timestamp=datetime.now().isoformat(),
            details={"evidently_report": output_path},
        )


class ModelDriftMonitor:
    """
    Monitor model performance drift over time.
    
    Tracks key metrics and detects degradation.
    """
    
    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        degradation_threshold: float = 0.05,  # 5% degradation
        primary_metric: str = "f1_macro",
    ):
        """
        Initialize monitor.
        
        Args:
            baseline_metrics: Baseline performance metrics
            degradation_threshold: Threshold for degradation detection
            primary_metric: Primary metric to track
        """
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        self.primary_metric = primary_metric
    
    def check_performance(
        self,
        current_metrics: Dict[str, float],
    ) -> DriftResult:
        """
        Check for performance degradation.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            DriftResult
        """
        baseline_value = self.baseline_metrics.get(self.primary_metric, 0)
        current_value = current_metrics.get(self.primary_metric, 0)
        
        if baseline_value == 0:
            degradation = 0
        else:
            degradation = (baseline_value - current_value) / baseline_value
        
        drift_detected = degradation > self.degradation_threshold
        
        # Calculate all metric degradations
        metric_changes = {}
        for metric, baseline in self.baseline_metrics.items():
            current = current_metrics.get(metric, 0)
            if baseline > 0:
                change = (current - baseline) / baseline
            else:
                change = 0
            metric_changes[metric] = {
                "baseline": baseline,
                "current": current,
                "change_pct": change * 100,
            }
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_score=degradation,
            drift_type="model",
            metrics={
                "primary_metric": self.primary_metric,
                "baseline": baseline_value,
                "current": current_value,
                "degradation_pct": degradation * 100,
            },
            threshold=self.degradation_threshold,
            timestamp=datetime.now().isoformat(),
            details={"metric_changes": metric_changes},
        )


class BiometricMonitor:
    """
    Complete monitoring solution for biometric pipeline.
    
    Combines data drift + model drift monitoring.
    """
    
    def __init__(
        self,
        config: Config,
        reference_features_path: Optional[str] = None,
        baseline_metrics_path: Optional[str] = None,
    ):
        """
        Initialize monitor.
        
        Args:
            config: Configuration object
            reference_features_path: Path to reference feature CSV
            baseline_metrics_path: Path to baseline metrics JSON
        """
        self.config = config
        self.reference_features_path = reference_features_path
        self.baseline_metrics_path = baseline_metrics_path
        
        self.reference_features = None
        self.baseline_metrics = None
        
        # Load reference data if available
        if reference_features_path and Path(reference_features_path).exists():
            self.reference_features = pd.read_csv(reference_features_path)
        
        if baseline_metrics_path and Path(baseline_metrics_path).exists():
            with open(baseline_metrics_path) as f:
                self.baseline_metrics = json.load(f)
    
    def extract_features_from_dataset(
        self,
        data_dir: str,
        output_path: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract features from dataset for drift analysis.
        
        Args:
            data_dir: Directory containing images
            output_path: Path to save features CSV
            sample_size: Max samples to process
            
        Returns:
            DataFrame with extracted features
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")
        
        from PIL import Image
        
        data_path = Path(data_dir)
        iris_dir = data_path / "iris"
        fp_dir = data_path / "fingerprint"
        
        features_list = []
        extractor = FeatureExtractor()
        
        iris_files = list(iris_dir.glob("*.bmp"))
        if sample_size:
            iris_files = iris_files[:sample_size]
        
        for iris_path in iris_files:
            try:
                # Load iris
                iris_img = np.array(Image.open(iris_path).convert("L"))
                iris_features = extractor.extract_basic_features(iris_img)
                iris_features = {f"iris_{k}": v for k, v in iris_features.items()}
                
                # Find matching fingerprint
                subject_id = iris_path.stem.split("_")[0]
                fp_pattern = f"{subject_id}_*.BMP"
                fp_files = list(fp_dir.glob(fp_pattern))
                
                if fp_files:
                    fp_img = np.array(Image.open(fp_files[0]).convert("L"))
                    fp_features = extractor.extract_basic_features(fp_img)
                    fp_features = {f"fingerprint_{k}": v for k, v in fp_features.items()}
                    
                    features = {**iris_features, **fp_features, "subject_id": subject_id}
                    features_list.append(features)
                    
            except Exception as e:
                logger.warning(f"Error processing {iris_path}: {e}")
        
        df = pd.DataFrame(features_list)
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Features saved to: {output_path}")
        
        return df
    
    def run_monitoring(
        self,
        current_data_dir: Optional[str] = None,
        current_features: Optional[pd.DataFrame] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        output_dir: str = "outputs/monitoring",
    ) -> MonitoringReport:
        """
        Run complete monitoring pipeline.
        
        Args:
            current_data_dir: Directory with current data
            current_features: Pre-extracted current features
            current_metrics: Current model performance metrics
            output_dir: Output directory for reports
            
        Returns:
            MonitoringReport
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data_drift_result = None
        model_drift_result = None
        recommendations = []
        
        # Data drift detection
        if self.reference_features is not None:
            if current_features is None and current_data_dir:
                current_features = self.extract_features_from_dataset(
                    current_data_dir,
                    sample_size=500,
                )
            
            if current_features is not None:
                data_monitor = DataDriftMonitor(
                    self.reference_features,
                    drift_threshold=0.1,
                )
                
                # Try Evidently first
                if EVIDENTLY_AVAILABLE:
                    data_drift_result = data_monitor.detect_drift_evidently(
                        current_features,
                        output_path=str(output_path / "data_drift_report.html"),
                    )
                else:
                    data_drift_result = data_monitor.detect_drift_statistical(
                        current_features
                    )
                
                if data_drift_result.drift_detected:
                    recommendations.append(
                        "Data drift detected! Consider retraining the model "
                        "with recent data."
                    )
        
        # Model drift detection
        if self.baseline_metrics is not None and current_metrics is not None:
            model_monitor = ModelDriftMonitor(
                self.baseline_metrics,
                degradation_threshold=0.05,
            )
            model_drift_result = model_monitor.check_performance(current_metrics)
            
            if model_drift_result.drift_detected:
                recommendations.append(
                    f"Model performance degraded by "
                    f"{model_drift_result.drift_score*100:.1f}%! "
                    "Trigger retraining pipeline."
                )
        
        # Determine overall status
        if (data_drift_result and data_drift_result.drift_detected) or \
           (model_drift_result and model_drift_result.drift_detected):
            if (model_drift_result and model_drift_result.drift_score > 0.1):
                overall_status = "critical"
            else:
                overall_status = "warning"
        else:
            overall_status = "healthy"
            recommendations.append("No significant drift detected. Model is healthy.")
        
        report = MonitoringReport(
            data_drift=data_drift_result,
            model_drift=model_drift_result,
            overall_status=overall_status,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
        )
        
        # Save report
        report.to_json(str(output_path / "monitoring_report.json"))
        
        return report


def create_reference_baseline(
    config: Config,
    output_dir: str = "outputs/monitoring/baseline",
) -> Tuple[str, str]:
    """
    Create reference data and baseline metrics for monitoring.
    
    Should be run after initial model training to establish baselines.
    
    Args:
        config: Configuration object
        output_dir: Output directory
        
    Returns:
        Tuple of (features_path, metrics_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    monitor = BiometricMonitor(config)
    
    # Extract reference features
    features_path = str(output_path / "reference_features.csv")
    features = monitor.extract_features_from_dataset(
        config.data.raw_dir,
        output_path=features_path,
    )
    logger.info(f"Reference features saved: {len(features)} samples")
    
    # Baseline metrics should come from model evaluation
    metrics_path = str(output_path / "baseline_metrics.json")
    
    # Check if we have evaluation metrics
    eval_metrics_path = Path("outputs/models/champion/metrics.json")
    if eval_metrics_path.exists():
        import shutil
        shutil.copy(eval_metrics_path, metrics_path)
        logger.info(f"Baseline metrics copied from champion model")
    else:
        # Create placeholder
        placeholder = {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "note": "Run model evaluation to populate baseline metrics"
        }
        with open(metrics_path, "w") as f:
            json.dump(placeholder, f, indent=2)
        logger.warning("No champion metrics found. Created placeholder baseline.")
    
    return features_path, metrics_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model and Data Drift Monitoring")
    parser.add_argument("--config", "-c", default="configs/config.yaml")
    parser.add_argument("--action", choices=["monitor", "baseline", "report"],
                       default="monitor", help="Action to perform")
    parser.add_argument("--data-dir", help="Current data directory to check")
    parser.add_argument("--reference", help="Reference features CSV path")
    parser.add_argument("--baseline-metrics", help="Baseline metrics JSON path")
    parser.add_argument("--output", "-o", default="outputs/monitoring")
    
    args = parser.parse_args()
    
    setup_logging()
    config = Config.from_yaml(args.config)
    
    if args.action == "baseline":
        logger.info("Creating reference baseline...")
        features_path, metrics_path = create_reference_baseline(config, args.output)
        print(f"\nBaseline created:")
        print(f"  Features: {features_path}")
        print(f"  Metrics: {metrics_path}")
        return
    
    if args.action == "report":
        # Generate HTML report with Evidently
        if not EVIDENTLY_AVAILABLE:
            print("Evidently not installed. Install: pip install evidently")
            return 1
        
        monitor = BiometricMonitor(
            config,
            reference_features_path=args.reference or "outputs/monitoring/baseline/reference_features.csv",
            baseline_metrics_path=args.baseline_metrics or "outputs/monitoring/baseline/baseline_metrics.json",
        )
        
        data_dir = args.data_dir or config.data.raw_dir
        report = monitor.run_monitoring(
            current_data_dir=data_dir,
            output_dir=args.output,
        )
        
        print(f"\nHTML report saved to: {args.output}/data_drift_report.html")
        return
    
    # Run monitoring
    monitor = BiometricMonitor(
        config,
        reference_features_path=args.reference or "outputs/monitoring/baseline/reference_features.csv",
        baseline_metrics_path=args.baseline_metrics or "outputs/monitoring/baseline/baseline_metrics.json",
    )
    
    data_dir = args.data_dir or config.data.raw_dir
    report = monitor.run_monitoring(
        current_data_dir=data_dir,
        output_dir=args.output,
    )
    
    # Print results
    print("\n" + "="*60)
    print("MONITORING REPORT")
    print("="*60)
    print(f"Status: {report.overall_status.upper()}")
    print(f"Timestamp: {report.timestamp}")
    
    if report.data_drift:
        print(f"\nData Drift:")
        print(f"  Detected: {report.data_drift.drift_detected}")
        print(f"  Score: {report.data_drift.drift_score:.2%}")
    
    if report.model_drift:
        print(f"\nModel Drift:")
        print(f"  Detected: {report.model_drift.drift_detected}")
        print(f"  Degradation: {report.model_drift.drift_score:.2%}")
    
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  • {rec}")
    
    print(f"\nFull report: {args.output}/monitoring_report.json")


if __name__ == "__main__":
    main()
