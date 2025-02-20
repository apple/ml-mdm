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
    channels = 64  # Number of channels
    batch_size = 2  # Batch size
    spatial_size = 8  # Spatial dimensions (H=W=8)
    cond_dim = 32  # Conditional dimension
    num_heads = 8  # Number of attention heads

    # Create model instances
    pytorch_attn = SelfAttention(
        channels=channels,
        num_heads=num_heads,
        cond_dim=cond_dim,
        use_attention_ffn=True,
    )
    mlx_attn = SelfAttention_MLX(  # Assuming this is your MLX class name
        channels=channels,
        num_heads=num_heads,
        cond_dim=cond_dim,
        use_attention_ffn=True,
    )

    # Set models to evaluation mode
    pytorch_attn.eval()
    mlx_attn.eval()

    # Create test inputs
    # Main input: [B, C, H, W]
    pytorch_input = torch.randn(batch_size, channels, spatial_size, spatial_size)
    # Conditional input: [B, seq_len, cond_dim]
    cond_seq_len = 4
    pytorch_cond = torch.randn(batch_size, cond_seq_len, cond_dim)
    # Conditional mask: [B, seq_len]
    pytorch_cond_mask = torch.ones(batch_size, cond_seq_len)

    # Test PyTorch version
    pytorch_output = pytorch_attn(
        pytorch_input, cond=pytorch_cond, cond_mask=pytorch_cond_mask
    )

    # Convert inputs to MLX format
    mlx_input = mx.array(pytorch_input.numpy())
    mlx_cond = mx.array(pytorch_cond.numpy())
    mlx_cond_mask = mx.array(pytorch_cond_mask.numpy())

    # Test MLX version
    mlx_output = mlx_attn.forward(mlx_input, cond=mlx_cond, cond_mask=mlx_cond_mask)

    # Validate outputs
    # Check shapes
    assert pytorch_output.shape == pytorch_input.shape, "PyTorch output shape mismatch"
    assert mlx_output.shape == mlx_input.shape, "MLX output shape mismatch"

    # Check numerical equivalence
    assert np.allclose(
        pytorch_output.detach().numpy(), np.array(mlx_output), atol=1e-5, rtol=1e-5
    ), "Outputs of PyTorch and MLX attention should match"

    # Test without conditional inputs
    pytorch_output_no_cond = pytorch_attn(pytorch_input)
    mlx_output_no_cond = mlx_attn.forward(mlx_input)

    assert np.allclose(
        pytorch_output_no_cond.detach().numpy(),
        np.array(mlx_output_no_cond),
        atol=1e-5,
        rtol=1e-5,
    ), "Outputs without conditioning should match"

    print("Self-attention test passed for both PyTorch and MLX!")
