# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import mlx.core as mx
import numpy as np
import torch

from ml_mdm.models.unet import MLP, SelfAttention
from ml_mdm.models.unet_mlx import MLP_MLX, SelfAttention_MLX

def test_pytorch_mlp():
    """
    Simple test for our MLP implementations
    """
    # Define parameters
    channels = 8  # Number of channels
    multiplier = 4  # Multiplier for hidden dimensions

    # Create a model instance
    pytorch_mlp = MLP(channels=channels, multiplier=multiplier)
    mlx_mlp = MLP_MLX(channels=channels, multiplier=multiplier)

    ## Start by testing pytorch version

    # Set model to evaluation mode
    pytorch_mlp.eval()

    # Create a dummy pytorch input tensor (batch size = 2, channels = 8)
    input_tensor = torch.randn(2, channels)

    # Pass the input through the model
    output = pytorch_mlp(input_tensor)

    # Assertions to validate the output shape and properties
    assert output.shape == input_tensor.shape, "Output shape mismatch"
    assert torch.allclose(
        output, input_tensor, atol=1e-5
    ), "Output should be close to input as the final layer is zero-initialized"

    ## now test mlx version

    # Convert the same input to MLX tensor
    mlx_tensor = mx.array(input_tensor.numpy())

    mlx_mlp.eval()

    mlx_output = mlx_mlp.forward(mlx_tensor)

    assert isinstance(mlx_output, mx.array)
    assert mlx_output.shape == input_tensor.shape, "MLX MLP: Output shape mismatch"

    # Validate numerical equivalence using numpy
    assert np.allclose(
        output.detach().numpy(), np.array(mlx_output), atol=1e-5
    ), "Outputs of PyTorch MLP and MLX MLP should match"

    print("Test passed for both PyTorch and MLX MLP!")


def test_pytorch_mlx_self_attention():
    """
    Test for feature parity between PyTorch and MLX implementations of SelfAttention.
    We'll test both the basic self-attention and conditional attention scenarios.
    """
    # Define test parameters
    channels = 64
    batch_size = 2
    spatial_size = 8
    cond_dim = 32
    num_heads = 8

    # ===== 1. Test WITH CONDITIONAL INPUT =====
    # Create models WITH conditional support
    pytorch_attn_with_cond = SelfAttention(
        channels=channels,
        num_heads=num_heads,
        cond_dim=cond_dim,  # Enable conditioning
        use_attention_ffn=True,
    )
    mlx_attn_with_cond = SelfAttention_MLX(
        channels=channels,
        num_heads=num_heads,
        cond_dim=cond_dim,
        use_attention_ffn=True,
    )

    # Create conditional inputs
    cond_seq_len = 4
    pytorch_cond = torch.randn(batch_size, cond_seq_len, cond_dim)
    pytorch_cond_mask = torch.ones(batch_size, cond_seq_len)
    mlx_cond = mx.array(pytorch_cond.numpy())
    mlx_cond_mask = mx.array(pytorch_cond_mask.numpy())

    # Run conditional tests
    pytorch_input = torch.randn(batch_size, channels, spatial_size, spatial_size)
    mlx_input = mx.array(pytorch_input.numpy())

    # PyTorch conditional forward
    pytorch_output_with_cond = pytorch_attn_with_cond(
        pytorch_input, cond=pytorch_cond, cond_mask=pytorch_cond_mask
    )
    # MLX conditional forward
    mlx_output_with_cond = mlx_attn_with_cond.forward(
        mlx_input, cond=mlx_cond, cond_mask=mlx_cond_mask
    )

    # ===== 2. Test WITHOUT CONDITIONAL INPUT =====
    # Create NEW models WITHOUT conditional support
    pytorch_attn_no_cond = SelfAttention(
        channels=channels,
        num_heads=num_heads,
        cond_dim=None,  
        use_attention_ffn=True,
    )
    mlx_attn_no_cond = SelfAttention_MLX(
        channels=channels,
        num_heads=num_heads,
        cond_dim=None,  
        use_attention_ffn=True,
    )

    # Run non-conditional tests
    pytorch_output_no_cond = pytorch_attn_no_cond(pytorch_input)
    mlx_output_no_cond = mlx_attn_no_cond.forward(mlx_input)

    # ===== Assertions =====
    # Check conditional outputs
    assert pytorch_output_with_cond.shape == pytorch_input.shape
    assert mlx_output_with_cond.shape == mlx_input.shape
    assert np.allclose(
        pytorch_output_with_cond.detach().numpy(),
        np.array(mlx_output_with_cond),
        atol=1e-5, rtol=1e-5
    ), "Outputs of PyTorch and MLX attention should match"

    # Check non-conditional outputs
    assert pytorch_output_no_cond.shape == pytorch_input.shape
    assert mlx_output_no_cond.shape == mlx_input.shape
    assert np.allclose(
        pytorch_output_no_cond.detach().numpy(),
        np.array(mlx_output_no_cond),
        atol=1e-5, rtol=1e-5
    ), "Outputs without conditioning should match"

    print("Self-attention test passed for both PyTorch and MLX!")