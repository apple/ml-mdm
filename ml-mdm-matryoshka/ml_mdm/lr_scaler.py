# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
from torch.optim.lr_scheduler import LambdaLR


class LRScaler:
    def __init__(self, scale=1.0):
        self._scale = scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def lr_lambda(self, current_step):
        current_step = max(1, current_step)
        if current_step < self.warmup_steps:
            return self._scale * float(current_step) / float(max(1, self.warmup_steps))
        # if current_step >= self.warmup_steps:
        #     return self._scale * float(self.warmup_steps) / current_step
        return self._scale

    def get_lr_scheduler(self, warmup_steps, optimizer):
        self.warmup_steps = warmup_steps
        return LambdaLR(optimizer, self.lr_lambda)
