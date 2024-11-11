# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import glob

from ml_mdm import config, diffusion
from ml_mdm.models import nested_unet, unet


def test_unet_in_registry():
    """Check that 'nested_unet' and 'unet' models are correctly registered in the Model Registry."""
    assert config.get_model("nested_unet") is not None
    assert config.get_model("unet") is not None


def test_unet_in_pipeline():
    """Check that 'nested_unet' and 'unet' models have corresponding pipelines defined."""
    assert config.get_pipeline("unet") is not None
    assert config.get_pipeline("nested_unet") is not None


def test_config_cc12m_64x64():
    """Check that the 'cc12m_64x64' configuration file loads successfully for all pipeline modes (trainer, sampler, demo)."""
    f = "configs/models/cc12m_64x64.yaml"
    f = "configs/models/cc12m_64x64.yaml"
    args = config.get_arguments(
        mode="trainer",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        mode="trainer",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        mode="sampler",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        mode="demo",
        additional_config_paths=[f],
    )
    assert args


def test_config_cc12m_256x256():
    """Check that the 'cc12m_256x256' configuration loads with 'nested_unet' as model in all modes (trainer, sampler, demo)."""
    f = "configs/models/cc12m_256x256.yaml"
    args = config.get_arguments(
        args=["--model=nested_unet"],
        mode="trainer",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        args=["--model=nested_unet"],
        mode="trainer",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        args=["--model=nested_unet"],
        mode="sampler",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        args=["--model=nested_unet"],
        mode="demo",
        additional_config_paths=[f],
    )
    assert args


def test_config_cc12m_1024x1024():
    """Check that the 'cc12m_1024x1024' configuration loads with 'nested2_unet' model in all modes (trainer, sampler, demo)."""
    f = "configs/models/cc12m_1024x1024.yaml"
    args = config.get_arguments(
        args=["--model=nested2_unet"],
        mode="trainer",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        args=["--model=nested2_unet"],
        mode="trainer",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        args=["--model=nested2_unet"],
        mode="sampler",
        additional_config_paths=[f],
    )
    assert args

    args = config.get_arguments(
        args=["--model=nested2_unet"],
        mode="demo",
        additional_config_paths=[f],
    )
    assert args
