# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import mlx.core as mx
import numpy as np
import torch

from ml_mdm.models.unet import MLP, SelfAttention1D, TemporalAttentionBlock
from ml_mdm.models.unet_mlx import (
    MLP_MLX,
    SelfAttention1D_MLX,
    TemporalAttentionBlock_MLX,
)


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


def test_self_attention_1d():
    # Define parameters
    channels = 8
    num_heads = 2
    seq_length = 16
    batch_size = 2

    # Create a model instance
    pytorch_attn = SelfAttention1D(channels=channels, num_heads=num_heads)
    mlx_attn = SelfAttention1D_MLX(channels=channels, num_heads=num_heads)

    # Set models to evaluation mode
    pytorch_attn.eval()
    mlx_attn.eval()

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, seq_length, channels)

    # Pass the input through the PyTorch model
    pytorch_output = pytorch_attn(input_tensor, mask=None)

    # Convert the input to MLX format
    mlx_input = mx.array(input_tensor.numpy())

    # Pass the input through the MLX model
    mlx_output = mlx_attn.forward(mlx_input, mask=None)

    # Assertions to validate the output shape and properties
    assert pytorch_output.shape == mlx_output.shape, "Output shape mismatch"
    assert np.allclose(
        pytorch_output.detach().numpy(), np.array(mlx_output), atol=1e-5
    ), "Outputs of PyTorch and MLX SelfAttention1D should match"

    print("Test passed for both PyTorch and MLX SelfAttention1D!")


def test_pytorch_mlx_temporal_attention_block():
    """
    Test for verifying parity between PyTorch and MLX implementations of TemporalAttentionBlock
    """
    # Define parameters
    channels = 8
    num_heads = 2
    batch_size = 2
    time_steps = 4
    height = 16
    width = 16

    # Create model instances
    pytorch_block = TemporalAttentionBlock(
        channels=channels, num_heads=num_heads, down=True
    )

    mlx_block = TemporalAttentionBlock_MLX(
        channels=channels, num_heads=num_heads, down=True
    )

    # Set models to evaluation mode
    pytorch_block.eval()
    mlx_block.eval()

    # Create dummy input tensors
    pytorch_input = torch.randn(batch_size * time_steps, channels, height, width)
    pytorch_temb = torch.randn(batch_size, channels)

    # Pass inputs through PyTorch model
    pytorch_output = pytorch_block(pytorch_input, pytorch_temb)

    # Convert to MLX format
    mlx_input = mx.array(pytorch_input.numpy())
    mlx_temb = mx.array(pytorch_temb.numpy())

    # Pass inputs through MLX model
    mlx_output = mlx_block.forward(mlx_input, mlx_temb)

    # print output tensors for debug
    print("pytorch_output tensor: ", pytorch_output)
    print("mlx_output tensor: ", mlx_output)

    # Assertions to validate the output
    assert pytorch_output.shape == tuple(mlx_output.shape), "Output shape mismatch"
    assert np.allclose(
        pytorch_output.detach().numpy(), np.array(mlx_output), rtol=1e-1, atol=1e-1
    ), "Outputs of PyTorch and MLX TemporalAttentionBlock should match"

    print("Test passed for both PyTorch and MLX TemporalAttentionBlock!")