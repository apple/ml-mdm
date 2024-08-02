# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
from ml_mdm import config


def test_load_flick_config():
    args = config.get_arguments(
        "",
        mode="demo",
        additional_config_paths=[f"configs/models/cc12m_64x64.yaml"],
    )
    assert args.reader_config.smaller_side_size == 64
