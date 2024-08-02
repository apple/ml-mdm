# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
import sys

import numpy as np


def print_args(args):
    command_str = f"python {sys.argv[0]} "
    for k, v in vars(args).items():
        command_str += f"\\\n\t {k}={v}"
    logging.info(command_str)
