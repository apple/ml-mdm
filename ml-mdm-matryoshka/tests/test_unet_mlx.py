# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import mlx.core as mx
import numpy as np
import torch

from ml_mdm.models.unet import MLP, ResNet, ResNetConfig
from ml_mdm.models.unet_mlx import MLP_MLX, ResNet_MLX, init_weights, zero_module_mlx


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

def test_pytorch_mlx_ResNet():
    """Test that PyTorch and MLX ResNet implementations produce matching outputs."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    
    # Define parameters
    batch_size = 2
    time_emb_channels = 32
    height = 16
    width = 16

    # Create config
    config = ResNetConfig(
        num_channels=64,
        output_channels=64,  # Match input channels for testing
        num_groups_norm=32,
        dropout=0.0,  # Set to 0 for deterministic comparison
        use_attention_ffn=False,
    )

    # Create model instances
    pytorch_resnet = ResNet(time_emb_channels=time_emb_channels, config=config)
    mlx_resnet = ResNet_MLX(time_emb_channels=time_emb_channels, config=config)

    # Initialize weights for MLX model
    init_weights(mlx_resnet.norm1)
    init_weights(mlx_resnet.conv1)
    init_weights(mlx_resnet.time_layer)
    init_weights(mlx_resnet.norm2)
    mlx_resnet.conv2 = zero_module_mlx(mlx_resnet.conv2)
    if hasattr(mlx_resnet, 'conv3'):
        init_weights(mlx_resnet.conv3)

    # Ensure weights have correct shapes for GroupNorm
    if hasattr(mlx_resnet.norm1, 'weight'):
        mlx_resnet.norm1.weight = mx.array(np.ones(config.num_channels))
        mlx_resnet.norm1.bias = mx.array(np.zeros(config.num_channels))
    if hasattr(mlx_resnet.norm2, 'weight'):
        mlx_resnet.norm2.weight = mx.array(np.ones(config.output_channels))
        mlx_resnet.norm2.bias = mx.array(np.zeros(config.output_channels))

    # Set both models to evaluation mode
    pytorch_resnet.eval()
    mlx_resnet.eval()

    # Create input tensors with same random seed for reproducibility
    torch.manual_seed(42)
    # Create input tensor with num_channels (64) channels
    x_torch = torch.randn(batch_size, config.num_channels, height, width)  # [2, 64, 16, 16]
    temb_torch = torch.randn(batch_size, time_emb_channels)  # [2, 32]

    # Get PyTorch output first
    with torch.no_grad():
        output_torch = pytorch_resnet(x_torch, temb_torch)

    # Convert inputs to MLX format (NCHW -> NHWC)
    x_numpy = x_torch.detach().numpy()
    x_numpy = np.transpose(x_numpy, (0, 2, 3, 1))  # NCHW -> NHWC
    x_mlx = mx.array(x_numpy)
    temb_mlx = mx.array(temb_torch.detach().numpy())

    # Debug shapes and intermediate values
    print("\nInput shapes:")
    print("PyTorch x (NCHW):", x_torch.shape)
    print("MLX x (NHWC):", x_mlx.shape)
    print("PyTorch temb:", temb_torch.shape)
    print("MLX temb:", temb_mlx.shape)

    # Debug intermediate values in MLX
    # Convert input to NCHW for MLX processing
    x_nchw = mx.transpose(x_mlx, [0, 3, 1, 2])
    output_mlx = mlx_resnet.forward(x_mlx, temb_mlx)

    # Convert MLX output to NCHW format for comparison
    output_mlx_numpy = np.array(output_mlx)
    output_mlx_numpy = np.transpose(output_mlx_numpy, (0, 3, 1, 2))  # NHWC -> NCHW

    # Compare outputs
    np.testing.assert_allclose(
        output_torch.detach().numpy(),
        output_mlx_numpy,
        rtol=1e-4,
        atol=1e-4,
    )
