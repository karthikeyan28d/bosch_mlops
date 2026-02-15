"""
PyTorch model architecture for multimodal biometric recognition.

This module provides:
- MultimodalBiometricModel: Main model with iris and fingerprint branches
- Modular branch architecture
- Multiple fusion strategies
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("biometric_mlops")


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm and activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding (default: kernel_size // 2)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ModalityBranch(nn.Module):
    """
    CNN branch for processing a single modality (iris or fingerprint).
    
    Args:
        in_channels: Input channels (1 for grayscale)
        conv_channels: List of channel sizes for conv layers
        embedding_dim: Output embedding dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = None,
        embedding_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        conv_channels = conv_channels or [32, 64, 128]
        
        # Build convolutional layers
        layers = []
        current_channels = in_channels
        
        for out_channels in conv_channels:
            layers.append(ConvBlock(current_channels, out_channels))
            layers.append(nn.MaxPool2d(2, 2))
            current_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate feature map size after conv layers
        # For 128x128 input with 3 pooling layers: 128 / 2^3 = 16
        self.feature_size = conv_channels[-1] * 16 * 16
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 1, H, W]
            
        Returns:
            Embedding tensor [B, embedding_dim]
        """
        features = self.features(x)
        embedding = self.fc(features)
        return embedding


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for multimodal embeddings.
    
    Learns attention weights to combine modality embeddings.
    """
    
    def __init__(self, embedding_dim: int, num_modalities: int = 2):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse embeddings with learned attention weights.
        
        Args:
            embeddings: List of [B, D] tensors
            
        Returns:
            Fused embedding [B, D]
        """
        # Stack embeddings: [B, num_modalities, D]
        stacked = torch.stack(embeddings, dim=1)
        
        # Concatenate for attention input: [B, num_modalities * D]
        concat = torch.cat(embeddings, dim=-1)
        
        # Compute attention weights: [B, num_modalities]
        weights = self.attention(concat)
        
        # Apply attention: [B, D]
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        
        return fused


class MultimodalBiometricModel(nn.Module):
    """
    Multimodal biometric recognition model.
    
    Combines iris and fingerprint modalities through configurable fusion.
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Embedding dimension for each branch
        dropout: Dropout rate
        fusion_method: Fusion strategy ("concat", "attention", "weighted_sum")
        conv_channels: Channel sizes for conv layers
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        fusion_method: str = "concat",
        conv_channels: List[int] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method
        
        conv_channels = conv_channels or [32, 64, 128]
        
        # Modality-specific branches
        self.iris_branch = ModalityBranch(
            in_channels=1,
            conv_channels=conv_channels,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        self.fingerprint_branch = ModalityBranch(
            in_channels=1,
            conv_channels=conv_channels,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Fusion layer
        if fusion_method == "concat":
            fused_dim = embedding_dim * 2
        elif fusion_method == "attention":
            self.attention_fusion = AttentionFusion(embedding_dim, num_modalities=2)
            fused_dim = embedding_dim
        elif fusion_method == "weighted_sum":
            # Learnable weights for each modality
            self.modality_weights = nn.Parameter(torch.ones(2) / 2)
            fused_dim = embedding_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Model initialized: classes={num_classes}, "
                   f"fusion={fusion_method}, embed_dim={embedding_dim}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        iris: torch.Tensor,
        fingerprint: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            iris: Iris images [B, 1, H, W]
            fingerprint: Fingerprint images [B, 1, H, W]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dict with 'logits' and optionally 'embeddings'
        """
        # Extract modality embeddings
        iris_emb = self.iris_branch(iris)
        fp_emb = self.fingerprint_branch(fingerprint)
        
        # Fuse embeddings
        if self.fusion_method == "concat":
            fused = torch.cat([iris_emb, fp_emb], dim=-1)
        elif self.fusion_method == "attention":
            fused = self.attention_fusion([iris_emb, fp_emb])
        elif self.fusion_method == "weighted_sum":
            weights = F.softmax(self.modality_weights, dim=0)
            fused = weights[0] * iris_emb + weights[1] * fp_emb
        
        # Classification
        logits = self.classifier(fused)
        
        output = {"logits": logits}
        
        if return_embeddings:
            output["iris_embedding"] = iris_emb
            output["fingerprint_embedding"] = fp_emb
            output["fused_embedding"] = fused
        
        return output
    
    def get_embeddings(
        self,
        iris: torch.Tensor,
        fingerprint: torch.Tensor
    ) -> torch.Tensor:
        """
        Get fused embeddings (for inference/retrieval).
        
        Args:
            iris: Iris images [B, 1, H, W]
            fingerprint: Fingerprint images [B, 1, H, W]
            
        Returns:
            Fused embeddings [B, embedding_dim]
        """
        output = self.forward(iris, fingerprint, return_embeddings=True)
        return output["fused_embedding"]
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_from_config(config, num_classes: int) -> MultimodalBiometricModel:
    """
    Create model from configuration.
    
    Args:
        config: Configuration object
        num_classes: Number of classes
        
    Returns:
        Initialized model
    """
    model_config = config.model
    
    # Extract conv channels
    conv_channels = None
    if hasattr(model_config, "iris_branch"):
        conv_layers = model_config.iris_branch.conv_layers
        conv_channels = [layer[0] for layer in conv_layers]
    
    model = MultimodalBiometricModel(
        num_classes=num_classes,
        embedding_dim=model_config.embedding_dim,
        dropout=model_config.dropout,
        fusion_method=model_config.fusion_method,
        conv_channels=conv_channels
    )
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = MultimodalBiometricModel(
        num_classes=100,
        embedding_dim=128,
        fusion_method="concat"
    )
    
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    iris = torch.randn(batch_size, 1, 128, 128)
    fingerprint = torch.randn(batch_size, 1, 128, 128)
    
    output = model(iris, fingerprint, return_embeddings=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Iris embedding shape: {output['iris_embedding'].shape}")
    print(f"Fused embedding shape: {output['fused_embedding'].shape}")
