"""
Unit tests for model module.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConvBlock:
    """Tests for ConvBlock class."""
    
    def test_conv_block_output_shape(self):
        """Test ConvBlock maintains spatial dimensions with padding."""
        from src.model import ConvBlock
        
        block = ConvBlock(1, 32, kernel_size=3)
        
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        
        assert out.shape == (2, 32, 64, 64)
    
    def test_conv_block_channels(self):
        """Test ConvBlock changes channels correctly."""
        from src.model import ConvBlock
        
        block = ConvBlock(16, 64)
        
        x = torch.randn(1, 16, 32, 32)
        out = block(x)
        
        assert out.shape[1] == 64


class TestModalityBranch:
    """Tests for ModalityBranch class."""
    
    def test_branch_output_shape(self):
        """Test branch produces correct embedding dimension."""
        from src.model import ModalityBranch
        
        branch = ModalityBranch(
            in_channels=1,
            conv_channels=[32, 64],
            embedding_dim=128
        )
        
        x = torch.randn(4, 1, 128, 128)
        out = branch(x)
        
        assert out.shape == (4, 128)
    
    def test_branch_different_embedding_dims(self):
        """Test branch works with different embedding dimensions."""
        from src.model import ModalityBranch
        
        for embed_dim in [64, 128, 256]:
            branch = ModalityBranch(embedding_dim=embed_dim)
            x = torch.randn(2, 1, 128, 128)
            out = branch(x)
            
            assert out.shape == (2, embed_dim)


class TestMultimodalBiometricModel:
    """Tests for MultimodalBiometricModel class."""
    
    def test_model_forward_concat_fusion(self):
        """Test forward pass with concat fusion."""
        from src.model import MultimodalBiometricModel
        
        model = MultimodalBiometricModel(
            num_classes=10,
            embedding_dim=64,
            fusion_method="concat"
        )
        
        iris = torch.randn(2, 1, 128, 128)
        fp = torch.randn(2, 1, 128, 128)
        
        output = model(iris, fp)
        
        assert "logits" in output
        assert output["logits"].shape == (2, 10)
    
    def test_model_forward_attention_fusion(self):
        """Test forward pass with attention fusion."""
        from src.model import MultimodalBiometricModel
        
        model = MultimodalBiometricModel(
            num_classes=10,
            embedding_dim=64,
            fusion_method="attention"
        )
        
        iris = torch.randn(2, 1, 128, 128)
        fp = torch.randn(2, 1, 128, 128)
        
        output = model(iris, fp)
        
        assert output["logits"].shape == (2, 10)
    
    def test_model_forward_weighted_sum_fusion(self):
        """Test forward pass with weighted sum fusion."""
        from src.model import MultimodalBiometricModel
        
        model = MultimodalBiometricModel(
            num_classes=10,
            embedding_dim=64,
            fusion_method="weighted_sum"
        )
        
        iris = torch.randn(2, 1, 128, 128)
        fp = torch.randn(2, 1, 128, 128)
        
        output = model(iris, fp)
        
        assert output["logits"].shape == (2, 10)
    
    def test_model_return_embeddings(self):
        """Test that embeddings are returned when requested."""
        from src.model import MultimodalBiometricModel
        
        model = MultimodalBiometricModel(
            num_classes=10,
            embedding_dim=64,
            fusion_method="concat"
        )
        
        iris = torch.randn(2, 1, 128, 128)
        fp = torch.randn(2, 1, 128, 128)
        
        output = model(iris, fp, return_embeddings=True)
        
        assert "iris_embedding" in output
        assert "fingerprint_embedding" in output
        assert "fused_embedding" in output
        
        assert output["iris_embedding"].shape == (2, 64)
        assert output["fingerprint_embedding"].shape == (2, 64)
    
    def test_model_get_embeddings(self):
        """Test get_embeddings method."""
        from src.model import MultimodalBiometricModel
        
        model = MultimodalBiometricModel(
            num_classes=10,
            embedding_dim=128,
            fusion_method="concat"
        )
        
        iris = torch.randn(4, 1, 128, 128)
        fp = torch.randn(4, 1, 128, 128)
        
        embeddings = model.get_embeddings(iris, fp)
        
        # Concat fusion doubles embedding dim
        assert embeddings.shape == (4, 256)
    
    def test_model_parameter_count(self):
        """Test parameter counting."""
        from src.model import MultimodalBiometricModel
        
        model = MultimodalBiometricModel(
            num_classes=10,
            embedding_dim=64
        )
        
        param_count = model.count_parameters()
        
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        from src.model import MultimodalBiometricModel
        
        model = MultimodalBiometricModel(
            num_classes=10,
            embedding_dim=64
        )
        
        iris = torch.randn(2, 1, 128, 128, requires_grad=True)
        fp = torch.randn(2, 1, 128, 128, requires_grad=True)
        
        output = model(iris, fp)
        loss = output["logits"].sum()
        loss.backward()
        
        # Check gradients exist
        assert iris.grad is not None
        assert fp.grad is not None
        
        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAttentionFusion:
    """Tests for AttentionFusion module."""
    
    def test_attention_output_shape(self):
        """Test attention fusion output shape."""
        from src.model import AttentionFusion
        
        fusion = AttentionFusion(embedding_dim=64, num_modalities=2)
        
        emb1 = torch.randn(4, 64)
        emb2 = torch.randn(4, 64)
        
        fused = fusion([emb1, emb2])
        
        assert fused.shape == (4, 64)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        from src.model import AttentionFusion
        
        fusion = AttentionFusion(embedding_dim=64, num_modalities=2)
        
        # Access attention network to check weights
        emb1 = torch.randn(4, 64)
        emb2 = torch.randn(4, 64)
        
        concat = torch.cat([emb1, emb2], dim=-1)
        weights = fusion.attention(concat)
        
        # Weights should sum to 1 along modality dimension
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)
