# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import mlx.core as mx
import numpy as np
import torch

from ml_mdm.models.unet import MLP, ResNet, ResNetConfig
from ml_mdm.models.unet_mlx import MLP_MLX, ResNet_MLX


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


def test_pytorch_ResNet():
    """
    Simple test for our ResNet implementations
    """
    # Define parameters
    batch_size = 2
    time_emb_channels = 32
    height = 16
    width = 16

    # Create config
    config = ResNetConfig(
        num_channels=64,
        output_channels=128,
        num_groups_norm=32,
        dropout=0.0,  # Set to 0 for deterministic comparison
        use_attention_ffn=False,
    )

    # Create model instances
    pytorch_resnet = ResNet(time_emb_channels=time_emb_channels, config=config)
    mlx_resnet = ResNet_MLX(time_emb_channels=time_emb_channels, config=config)

    # Set both models to evaluation mode
    pytorch_resnet.eval()
    mlx_resnet.eval()

    # Create a dummy pytorch input tensor (batch size = 2, channels = 64, height, width = 16)
    x_torch = torch.randn(batch_size, config.num_channels, height, width)
    temb_torch = torch.randn(batch_size, time_emb_channels)

    # pass the input thorugh the model
    output_torch, activations_torch = pytorch_resnet(x_torch, temb_torch)

    # Convert inputs to MLX tensors
    x_mlx = mx.array(x_torch.numpy())
    temb_mlx = mx.array(temb_torch.numpy())

    # Get MLX output
    output_mlx, activations_mlx = mlx_resnet(x_mlx, temb_mlx)

    # Verify outputs match
    assert np.allclose(
        output_torch.detach().numpy(), np.array(output_mlx), atol=1e-5
    ), "PyTorch and MLX ResNet outputs should match"

    print("Test passed for ResNet implementations!")
