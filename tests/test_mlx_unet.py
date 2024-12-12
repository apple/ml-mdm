from ml_mdm.models.unet import MLP
from ml_mdm.models.mlx_unet import MLX_MLP
import mlx.core as mx
import torch
import numpy as np

def test_pytorch_mlp():
    channels = 8
    multiplier = 4

    # Inicializar os modelos
    pytorch_mlp = MLP(channels=channels, multiplier=multiplier)
    mlx_mlp = MLX_MLP(channels=channels, multiplier=multiplier)
    
    # Definir os modelos em modo de avaliação
    pytorch_mlp.eval()
    mlx_mlp.eval()

    # Criar um tensor de entrada dummy para PyTorch (batch size = 2, channels = 8)
    input_tensor = torch.randn(2, channels)

    # Passar a entrada pelo modelo PyTorch
    output = pytorch_mlp(input_tensor)

    # Verificar a forma do output
    assert output.shape == input_tensor.shape, "Output shape mismatch"

    # Verificar se o output está próximo da entrada
    assert torch.allclose(
        output, input_tensor, atol=1e-5
    ), "Output should be close to input as the final layer is zero-initialized"

    # Converter a mesma entrada para um tensor MLX
    mlx_tensor = mx.array(input_tensor.numpy())

    # Passar a entrada pelo modelo MLX
    mlx_output = mlx_mlp.forward(mlx_tensor)

    # Avaliar o objeto MLX para garantir que a computação foi realizada
    mx.eval(mlx_output)
    # Verificar se o output MLX é uma instância de mx.array
    assert isinstance(mlx_output, mx.array), "MLX output is not an instance of mx.array"

    # Verificar a forma do output MLX
    assert mlx_output.shape == input_tensor.shape, "MLX MLP: Output shape mismatch"

    # Converter os outputs para arrays NumPy para comparação
    output_numpy = output.cpu().detach().numpy()
    mlx_output_numpy = np.array(mlx_output)

    # Verificar se as formas dos arrays são iguais
    assert output_numpy.shape == mlx_output_numpy.shape, (
        f"Shapes do not match: output {output_numpy.shape}, mlx_output {mlx_output_numpy.shape}"
    )

    # Verificar se há valores NaN ou Inf nos arrays
    assert not np.isnan(output_numpy).any(), "NaN detected in output"
    assert not np.isinf(output_numpy).any(), "Inf detected in output"
    assert not np.isnan(mlx_output_numpy).any(), "NaN detected in mlx_output"
    assert not np.isinf(mlx_output_numpy).any(), "Inf detected in mlx_output"

    # Comparar os outputs dos dois modelos
    assert np.allclose(
        output_numpy, mlx_output_numpy, atol=1e-5
    ), "Outputs of PyTorch MLP and MLX MLP should match"

    print("Test passed for both PyTorch and MLX MLP!")
