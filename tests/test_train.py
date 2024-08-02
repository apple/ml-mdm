# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import os
import shlex

import pytest

from ml_mdm import config
from ml_mdm.clis import train_parallel

small_arguments = """
ml_mdm/clis/train_parallel.py --file-list=tests/test_files/sample_training_0.tsv --multinode=0	--output-dir=outputs \
    --text-model=google/flan-t5-small \
    --num_diffusion_steps=1 \
    --model_output_scale=0 \
	--num-training-steps=1 \
    --model_config_file="configs/models/cc12m_64x64.yaml"
    --fp16=0
"""


@pytest.mark.skip(
    reason="more effective to test this with torchrun, just here for documentation"
)
def test_small():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    args = config.get_arguments(args=shlex.split(small_arguments), mode="trainer")
    train_parallel.main(args)
    assert os.path.isfile("outputs/vis_model_000100.pth")
