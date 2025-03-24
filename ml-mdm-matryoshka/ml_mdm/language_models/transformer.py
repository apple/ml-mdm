# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
""" Deprecated -- but needed for loading older checkpoints. """
from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    name: str = "transformer_config"
