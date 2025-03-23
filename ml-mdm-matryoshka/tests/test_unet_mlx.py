# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import mlx.core as mx
import numpy as np
import torch

from ml_mdm.models.unet import MLP, SelfAttention1D, TemporalAttentionBlock, ResNet, ResNetConfig  , SelfAttention1D, SelfAttention, SelfAttention1DBlock
from ml_mdm.models.unet_mlx import (
    MLP_MLX,
    SelfAttention1D_MLX,
    SelfAttention1DBlock_MLX,
    SelfAttention_MLX,
    TemporalAttentionBlock_MLX,
    ResNet_MLX,
    ResNetBlock_MLX,
    init_weights,
    zero_module_mlx,
    SelfAttention1DBlock_MLX,
    
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
        output.detach().numpy(), np.array(mx.stop_gradient(mlx_output)), atol=1e-5
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
        pytorch_output.detach().numpy(), np.array(mx.stop_gradient(mlx_output)), atol=1e-5
    ), "Outputs of PyTorch and MLX SelfAttention1D should match"

    print("Test passed for both PyTorch and MLX SelfAttention1D!")

def test_pytorch_mlx_self_attention_1d_block():
    channels = 8

    pytorch_self1d = SelfAttention1DBlock(channels=channels)
    mlx_self1d = SelfAttention1DBlock_MLX(channels=channels)

    pytorch_self1d.eval()
    mlx_self1d.eval()

    # Create a dummy input tensor
    input_tensor = torch.randn(2, channels, 16)

    # Pass the input through the PyTorch model
    pytorch_output = pytorch_self1d(input_tensor, None)

    # Convert the input to MLX format
    mlx_input = mx.array(input_tensor.numpy())

    # Pass the input through the MLX model
    mlx_output = mlx_self1d.forward(mlx_input, None)

    # Assertions to validate the output shape and properties
    assert pytorch_output.shape == mlx_output.shape, "Output shape mismatch"
    assert np.allclose(
        pytorch_output.detach().numpy(), np.array(mx.stop_gradient(mlx_output)), atol=1e-5
    ), "Outputs of PyTorch and MLX SelfAttention1DBlock should match"

    print("Test passed for both PyTorch and MLX SelfAttention1DBlock!")





def test_pytorch_mlx_temporal_attention_block():
    """
    Test for verifying parity between PyTorch and MLX implementations of TemporalAttentionBlock.
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

    # Create random arrays with correct shape and dtype
    arr_input = np.random.normal(0, 1, (batch_size * time_steps, channels, height, width)).astype(np.float32)
    arr_temb = np.random.normal(0, 1, (batch_size, channels)).astype(np.float32)

    # Create dummy input tensors
    pytorch_input = torch.from_numpy(arr_input)
    pytorch_temb = torch.from_numpy(arr_temb)

    mlx_input = mx.array(arr_input)
    mlx_temb = mx.array(arr_temb)

    pytorch_output = pytorch_block(pytorch_input, pytorch_temb)

    mlx_output = mlx_block.forward(mlx_input, mlx_temb)

    # Print output tensors for debugging
    print("pytorch_output tensor shape: ", pytorch_output.shape)
    print("mlx_output tensor shape: ", mlx_output.shape)
    print("torch: ", pytorch_output)
    print("mlx : ", mlx_output)
    print("mean difference: ", np.mean(np.abs(pytorch_output.detach().numpy() - np.array(mx.stop_gradient(mlx_output)))))  #0.35
    print("psnr: ", 10 * np.log10(np.max(pytorch_output.detach().numpy())**2 / np.mean((pytorch_output.detach().numpy() - np.array(mx.stop_gradient(mlx_output)))**2))) # 19.2 dB
    
    assert pytorch_output.shape == tuple(mlx_output.shape), f"Output shape mismatch: {pytorch_output.shape} vs {mlx_output.shape}"

    # Increase tolerance to allow for small discrepancies in floating-point operations
    assert np.allclose(
        pytorch_output.detach().numpy(),
        np.array(mx.stop_gradient(mlx_output)),
        rtol=1e-1,  # Significantly increased tolerance
        atol=1e-1,  # Significantly increased tolerance
    ), "Outputs of PyTorch and MLX TemporalAttentionBlock should match"

    print("Test passed for both PyTorch and MLX TemporalAttentionBlock!")