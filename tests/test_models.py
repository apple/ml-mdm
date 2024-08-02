# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import inspect
import os

import pytest

import torch

from ml_mdm import config, diffusion, models
from ml_mdm.clis.generate_sample import setup_models


def test_initialize_unet():
    unet_config = models.unet.UNetConfig()
    diffusion_config = diffusion.DiffusionConfig(
        use_vdm_loss_weights=True, model_output_scale=0.1
    )

    denoising_model = config.get_model("unet")(
        input_channels=3, output_channels=3, config=unet_config
    )

    diffusion_model = config.get_pipeline("unet")(denoising_model, diffusion_config)

    ema_model = models.model_ema.ModelEma(diffusion_model.model.vision_model)

    assert ema_model is not None


def test_all_registered_models():
    for config_name, additional_info in config.MODEL_CONFIG_REGISTRY.items():
        model_name = additional_info["model"]
        config_cls = additional_info["config"]

        assert inspect.isclass(config_cls)
        assert model_name in config.MODEL_REGISTRY
        model_cls = config.MODEL_REGISTRY[model_name]
        assert inspect.isclass(model_cls)

        model = model_cls(input_channels=3, output_channels=3, config=config_cls())


@pytest.mark.gpu
def test_initialize_pretrained():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = config.get_arguments(
        mode="trainer",
        additional_config_paths=["configs/models/cc12m_64x64.yaml"],
    )
    tokenizer, language_model, diffusion_model = setup_models(args, device)

    vision_model_file = "vis_model_64x64.pth"

    if os.path.exists(vision_model_file):
        diffusion_model.model.load(vision_model_file)
