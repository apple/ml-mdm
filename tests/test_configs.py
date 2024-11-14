# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import glob

from ml_mdm import config, diffusion
from ml_mdm.models import nested_unet, unet


def test_unet_in_registry():
    assert config.get_model("nested_unet") is not None
    assert config.get_model("unet") is not None


def test_unet_in_pipeline():
    assert config.get_pipeline("unet") is not None
    assert config.get_pipeline("nested_unet") is not None


def test_config_cc12m_64x64():
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