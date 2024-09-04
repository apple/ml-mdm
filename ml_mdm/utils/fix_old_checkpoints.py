# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import sys

from ml_mdm import (
    config,
    diffusion,
    distributed,
    language_models,
    models,
    reader,
    samplers,
)


def mimic_old_modules():
    sys.modules["language_models"] = language_models
    sys.modules["reader"] = reader
    sys.modules["config"] = config
    sys.modules["models"] = models
    sys.modules["diffusion"] = diffusion
    sys.modules["samplers"] = samplers
    sys.modules["distributed"] = distributed
