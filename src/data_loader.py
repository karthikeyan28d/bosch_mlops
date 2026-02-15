"""
Data loading and dataset abstraction for multimodal biometric data.

This module provides:
- BiometricDataset: PyTorch Dataset for iris + fingerprint images
- DataLoader creation with proper collation
- Efficient lazy loading and caching
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger("biometric_mlops")


class BiometricDataset(Dataset):
    """
    PyTorch Dataset for multimodal biometric data (Iris + Fingerprint).
    
    Handles:
    - Paired loading of iris and fingerprint images
    - On-the-fly transformations
    - Label extraction from filenames
    - Memory-efficient lazy loading
    
    Dataset structure expected:
        data/raw/iris/001_L_01.bmp
        data/raw/fingerprint/001_L_thumb_01.BMP
        
    Where 001 is the subject ID (label).
    
    Args:
        data_dir: Root directory containing 'iris' and 'fingerprint' folders
        transform: Optional transforms to apply to images
        image_size: Target image size (height, width)
        modalities: List of modalities to load ['iris', 'fingerprint']
        cache_labels: Whether to cache label mappings
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (128, 128),
        modalities: List[str] = None,
        cache_labels: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.modalities = modalities or ["iris", "fingerprint"]
        
        # Default transform if none provided
        self.transform = transform or self._default_transform()
        
        # Discover and pair samples
        self.samples = self._discover_samples()
        
        # Build label mapping
        self.label_to_idx, self.idx_to_label = self._build_label_mapping()
        self.num_classes = len(self.label_to_idx)
        
        logger.info(f"Loaded {len(self.samples)} samples with {self.num_classes} classes")
    
    def _default_transform(self) -> Callable:
        """Default image transformation pipeline."""
        return T.Compose([
            T.Resize(self.image_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _discover_samples(self) -> List[Dict[str, Path]]:
        """
        Discover and pair iris-fingerprint samples by subject ID.
        
        Returns:
            List of dicts with modality paths and subject_id
        """
        samples = []
        
        # Get all iris images and extract subject IDs
        iris_dir = self.data_dir / "iris"
        fingerprint_dir = self.data_dir / "fingerprint"
        
        if not iris_dir.exists() or not fingerprint_dir.exists():
            logger.warning(f"Data directories not found: {iris_dir} or {fingerprint_dir}")
            return samples
        
        # Pattern to extract subject ID (first 3 digits)
        subject_pattern = re.compile(r"^(\d{3})")
        
        # Group files by subject
        iris_by_subject: Dict[str, List[Path]] = {}
        fp_by_subject: Dict[str, List[Path]] = {}
        
        # Collect iris images
        for img_path in iris_dir.glob("*.bmp"):
            match = subject_pattern.match(img_path.name)
            if match:
                subject_id = match.group(1)
                iris_by_subject.setdefault(subject_id, []).append(img_path)
        
        # Collect fingerprint images
        for img_path in fingerprint_dir.glob("*.BMP"):
            match = subject_pattern.match(img_path.name)
            if match:
                subject_id = match.group(1)
                fp_by_subject.setdefault(subject_id, []).append(img_path)
        
        # Also check lowercase extension
        for img_path in fingerprint_dir.glob("*.bmp"):
            match = subject_pattern.match(img_path.name)
            if match:
                subject_id = match.group(1)
                fp_by_subject.setdefault(subject_id, []).append(img_path)
        
        # Create paired samples (all combinations for each subject)
        common_subjects = set(iris_by_subject.keys()) & set(fp_by_subject.keys())
        
        for subject_id in sorted(common_subjects):
            iris_images = iris_by_subject[subject_id]
            fp_images = fp_by_subject[subject_id]
            
            # Create pairs (can use all combinations or match by index)
            for iris_path in iris_images:
                for fp_path in fp_images:
                    samples.append({
                        "subject_id": subject_id,
                        "iris_path": iris_path,
                        "fingerprint_path": fp_path
                    })
        
        return samples
    
    def _build_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build bidirectional label mappings."""
        unique_labels = sorted(set(s["subject_id"] for s in self.samples))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        return label_to_idx, idx_to_label
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load image from disk."""
        return Image.open(path).convert("L")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dict with 'iris', 'fingerprint' tensors and 'label' int
        """
        sample = self.samples[idx]
        
        # Load images
        iris_img = self._load_image(sample["iris_path"])
        fp_img = self._load_image(sample["fingerprint_path"])
        
        # Apply transforms
        iris_tensor = self.transform(iris_img)
        fp_tensor = self.transform(fp_img)
        
        # Get label
        label = self.label_to_idx[sample["subject_id"]]
        
        return {
            "iris": iris_tensor,
            "fingerprint": fp_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": sample["subject_id"]
        }


class ProcessedBiometricDataset(Dataset):
    """
    Dataset for loading preprocessed data from Parquet/Arrow files.
    
    Used after parallel preprocessing to load from cached features.
    """
    
    def __init__(
        self,
        parquet_path: Union[str, Path],
        transform: Optional[Callable] = None
    ):
        self.parquet_path = Path(parquet_path)
        
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow required for ProcessedBiometricDataset")
        
        # Load metadata
        self.table = pq.read_table(self.parquet_path)
        self.df = self.table.to_pandas()
        
        # Build label mapping
        unique_labels = sorted(self.df["subject_id"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        self.num_classes = len(self.label_to_idx)
        
        self.transform = transform
        
        logger.info(f"Loaded {len(self.df)} preprocessed samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load preprocessed features (stored as numpy arrays)
        iris_features = torch.tensor(
            np.frombuffer(row["iris_features"], dtype=np.float32)
        ).reshape(1, 128, 128)
        
        fp_features = torch.tensor(
            np.frombuffer(row["fingerprint_features"], dtype=np.float32)
        ).reshape(1, 128, 128)
        
        label = self.label_to_idx[row["subject_id"]]
        
        return {
            "iris": iris_features,
            "fingerprint": fp_features,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": row["subject_id"]
        }


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    image_size: Tuple[int, int] = (128, 128),
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        split_ratios: (train, val, test) ratios
        image_size: Target image size
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = BiometricDataset(
        data_dir=data_dir,
        image_size=image_size
    )
    
    # Calculate split sizes
    total = len(full_dataset)
    train_size = int(total * split_ratios[0])
    val_size = int(total * split_ratios[1])
    test_size = total - train_size - val_size
    
    # Create splits with reproducible seed
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, "
                f"val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, full_dataset.num_classes


def collate_multimodal(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multimodal batches.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Batched dict with stacked tensors
    """
    return {
        "iris": torch.stack([b["iris"] for b in batch]),
        "fingerprint": torch.stack([b["fingerprint"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "subject_id": [b["subject_id"] for b in batch]
    }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset creation
    dataset = BiometricDataset("data/raw")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Iris shape: {sample['iris'].shape}")
        print(f"Fingerprint shape: {sample['fingerprint'].shape}")
