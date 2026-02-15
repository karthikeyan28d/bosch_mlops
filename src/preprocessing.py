"""
Parallel preprocessing for multimodal biometric data.

This module provides:
- Ray-based parallel image preprocessing
- Multiprocessing fallback
- Efficient batch processing with PyArrow output
- Performance benchmarking utilities
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
from PIL import Image

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger("biometric_mlops")


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing run."""
    total_samples: int
    processed_samples: int
    failed_samples: int
    processing_time: float
    throughput: float  # samples per second
    
    def __str__(self) -> str:
        return (
            f"Preprocessing Stats:\n"
            f"  Total: {self.total_samples}\n"
            f"  Processed: {self.processed_samples}\n"
            f"  Failed: {self.failed_samples}\n"
            f"  Time: {self.processing_time:.2f}s\n"
            f"  Throughput: {self.throughput:.2f} samples/sec"
        )


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (128, 128),
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Preprocess a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target (height, width)
        normalize: Whether to normalize to [-1, 1]
        
    Returns:
        Preprocessed image as numpy array, or None if failed
    """
    try:
        # Load and convert to grayscale
        img = Image.open(image_path).convert("L")
        
        # Resize
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        
        # Convert to numpy
        arr = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1]
        if normalize:
            arr = (arr / 127.5) - 1.0
        else:
            arr = arr / 255.0
        
        return arr
        
    except Exception as e:
        logger.warning(f"Failed to process {image_path}: {e}")
        return None


def preprocess_sample(
    sample: Dict,
    target_size: Tuple[int, int] = (128, 128)
) -> Optional[Dict]:
    """
    Preprocess a single multimodal sample.
    
    Args:
        sample: Dict with iris_path, fingerprint_path, subject_id
        target_size: Target image size
        
    Returns:
        Dict with preprocessed features or None if failed
    """
    iris_features = preprocess_image(sample["iris_path"], target_size)
    fp_features = preprocess_image(sample["fingerprint_path"], target_size)
    
    if iris_features is None or fp_features is None:
        return None
    
    return {
        "subject_id": sample["subject_id"],
        "iris_features": iris_features.tobytes(),
        "fingerprint_features": fp_features.tobytes(),
        "iris_path": str(sample["iris_path"]),
        "fingerprint_path": str(sample["fingerprint_path"])
    }


class ParallelPreprocessor:
    """
    Parallel preprocessing engine supporting Ray and multiprocessing.
    
    Features:
    - Automatic backend selection (Ray if available)
    - Configurable worker count
    - Progress tracking
    - Error handling with fallback
    
    Args:
        backend: Processing backend ("ray", "multiprocessing", "sequential")
        num_workers: Number of parallel workers (None = auto)
        target_size: Target image size
        batch_size: Batch size for processing
    """
    
    def __init__(
        self,
        backend: str = "ray",
        num_workers: Optional[int] = None,
        target_size: Tuple[int, int] = (128, 128),
        batch_size: int = 100
    ):
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Determine number of workers
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        
        # Select backend
        if backend == "ray" and RAY_AVAILABLE:
            self.backend = "ray"
        elif backend == "multiprocessing":
            self.backend = "multiprocessing"
        else:
            self.backend = "sequential"
            
        logger.info(f"Preprocessor initialized: backend={self.backend}, "
                   f"workers={self.num_workers}")
    
    def process(
        self,
        samples: List[Dict],
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[List[Dict], PreprocessingStats]:
        """
        Process samples in parallel.
        
        Args:
            samples: List of sample dicts with paths
            output_path: Optional path to save processed data
            
        Returns:
            Tuple of (processed_samples, stats)
        """
        start_time = time.time()
        
        if self.backend == "ray":
            results = self._process_ray(samples)
        elif self.backend == "multiprocessing":
            results = self._process_multiprocessing(samples)
        else:
            results = self._process_sequential(samples)
        
        # Filter out failed samples
        processed = [r for r in results if r is not None]
        
        processing_time = time.time() - start_time
        stats = PreprocessingStats(
            total_samples=len(samples),
            processed_samples=len(processed),
            failed_samples=len(samples) - len(processed),
            processing_time=processing_time,
            throughput=len(processed) / processing_time if processing_time > 0 else 0
        )
        
        # Save to parquet if output path provided
        if output_path and processed and PYARROW_AVAILABLE:
            self._save_parquet(processed, output_path)
        
        logger.info(str(stats))
        return processed, stats
    
    def _process_ray(self, samples: List[Dict]) -> List[Optional[Dict]]:
        """Process using Ray."""
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_workers, ignore_reinit_error=True)
        
        @ray.remote
        def process_remote(sample, target_size):
            return preprocess_sample(sample, target_size)
        
        # Submit all tasks
        futures = [
            process_remote.remote(sample, self.target_size) 
            for sample in samples
        ]
        
        # Collect results with progress
        results = []
        for i in range(0, len(futures), self.batch_size):
            batch = futures[i:i + self.batch_size]
            batch_results = ray.get(batch)
            results.extend(batch_results)
            
            progress = min(i + self.batch_size, len(futures))
            logger.info(f"Progress: {progress}/{len(futures)}")
        
        return results
    
    def _process_multiprocessing(self, samples: List[Dict]) -> List[Optional[Dict]]:
        """Process using multiprocessing."""
        from functools import partial
        
        process_fn = partial(preprocess_sample, target_size=self.target_size)
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_fn, samples, chunksize=self.batch_size))
        
        return results
    
    def _process_sequential(self, samples: List[Dict]) -> List[Optional[Dict]]:
        """Process sequentially (for debugging or small datasets)."""
        results = []
        for i, sample in enumerate(samples):
            result = preprocess_sample(sample, self.target_size)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(samples)}")
        
        return results
    
    def _save_parquet(self, data: List[Dict], output_path: Union[str, Path]) -> None:
        """Save processed data to Parquet file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PyArrow table
        table = pa.Table.from_pylist(data)
        
        # Write to parquet
        pq.write_table(table, output_path, compression="snappy")
        
        logger.info(f"Saved processed data to {output_path}")


def benchmark_preprocessing(
    data_dir: Union[str, Path],
    target_size: Tuple[int, int] = (128, 128),
    backends: List[str] = None
) -> Dict[str, PreprocessingStats]:
    """
    Benchmark different preprocessing backends.
    
    Args:
        data_dir: Root data directory
        target_size: Target image size
        backends: List of backends to test
        
    Returns:
        Dict mapping backend name to stats
    """
    from .data_loader import BiometricDataset
    
    # Get samples
    dataset = BiometricDataset(data_dir, image_size=target_size)
    samples = dataset.samples
    
    if not samples:
        logger.warning("No samples found for benchmarking")
        return {}
    
    backends = backends or ["sequential", "multiprocessing"]
    if RAY_AVAILABLE:
        backends.append("ray")
    
    results = {}
    
    for backend in backends:
        logger.info(f"\nBenchmarking {backend}...")
        
        preprocessor = ParallelPreprocessor(
            backend=backend,
            target_size=target_size
        )
        
        _, stats = preprocessor.process(samples)
        results[backend] = stats
    
    # Print comparison
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    for backend, stats in results.items():
        print(f"\n{backend.upper()}:")
        print(f"  Time: {stats.processing_time:.2f}s")
        print(f"  Throughput: {stats.throughput:.2f} samples/sec")
    
    return results


def preprocess_from_config(config_path: Union[str, Path]) -> None:
    """
    Run preprocessing from configuration file.
    
    Args:
        config_path: Path to config YAML file
    """
    from .utils import Config
    from .data_loader import BiometricDataset
    
    config = Config.from_yaml(config_path)
    
    # Get data paths
    raw_dir = Path(config.data.raw_dir)
    processed_dir = Path(config.data.processed_dir)
    
    # Get preprocessing settings
    backend = config.preprocessing.backend
    num_workers = config.preprocessing.get("num_workers")
    target_size = tuple(config.data.image.size)
    batch_size = config.preprocessing.batch_size
    
    # Create dataset to discover samples
    dataset = BiometricDataset(raw_dir, image_size=target_size)
    
    if not dataset.samples:
        logger.error("No samples found in data directory")
        return
    
    # Create preprocessor and process
    preprocessor = ParallelPreprocessor(
        backend=backend,
        num_workers=num_workers,
        target_size=target_size,
        batch_size=batch_size
    )
    
    output_path = processed_dir / "processed_data.parquet"
    processed, stats = preprocessor.process(dataset.samples, output_path)
    
    print(f"\nPreprocessing complete!")
    print(stats)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess biometric data")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark comparison")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                       help="Data directory for benchmark")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.benchmark:
        benchmark_preprocessing(args.data_dir)
    else:
        preprocess_from_config(args.config)
