from ml_mdm.models.unet import MLP
from ml_mdm.models.mlx_unet import MLX_MLP
import mlx.core as mx
import torch
import numpy as np

def test_pytorch_mlp():

    """"Function to test the PyTorch and MLX MLP models"""
    
    channels = 8
    multiplier = 4

    pytorch_mlp = MLP(channels=channels, multiplier=multiplier)
    mlx_mlp = MLX_MLP(channels=channels, multiplier=multiplier)
    
    pytorch_mlp.eval()
    mlx_mlp.eval()

    input_tensor = torch.randn(2, channels)

    output = pytorch_mlp(input_tensor)

    assert output.shape == input_tensor.shape, "Output shape mismatch"

    assert torch.allclose(
        output, input_tensor, atol=1e-5
    ), "Output should be close to input as the final layer is zero-initialized"

    mlx_tensor = mx.array(input_tensor.numpy())

    mlx_output = mlx_mlp.forward(mlx_tensor)

    mx.eval(mlx_output)
    assert isinstance(mlx_output, mx.array), "MLX output is not an instance of mx.array"

    assert mlx_output.shape == input_tensor.shape, "MLX MLP: Output shape mismatch"

    output_numpy = output.cpu().detach().numpy()
    mlx_output_numpy = np.array(mlx_output)

    assert output_numpy.shape == mlx_output_numpy.shape, (
        f"Shapes do not match: output {output_numpy.shape}, mlx_output {mlx_output_numpy.shape}"
    )

    assert not np.isnan(output_numpy).any(), "NaN detected in output"
    assert not np.isinf(output_numpy).any(), "Inf detected in output"
    assert not np.isnan(mlx_output_numpy).any(), "NaN detected in mlx_output"
    assert not np.isinf(mlx_output_numpy).any(), "Inf detected in mlx_output"

    assert np.allclose(
        output_numpy, mlx_output_numpy, atol=1e-5
    ), "Outputs of PyTorch MLP and MLX MLP should match"

    print("Test passed for both PyTorch and MLX MLP!")
