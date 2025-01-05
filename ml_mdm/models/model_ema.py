# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
from copy import deepcopy

import torch
import torch.nn as nn

from ml_mdm.utils import fix_old_checkpoints


class ModelEma(nn.Module):
    def __init__(self, model, decay: float=0.9999, warmup_steps: int = 0, device: torch.device =None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
        self.warmup_steps = warmup_steps
        self.counter = 0

    def update(self, model):
        decay = (self.counter >= self.warmup_steps) * self.decay
        self.counter += 1
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.module.state_dict().items():
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.mul_(decay).add_(model_v, alpha=(1.0 - decay))

    def save(self, fname: str, other_items=None):
        logging.info(f"Saving EMA model file: {fname}")
        checkpoint = {"state_dict": self.module.state_dict()}
        if other_items is not None:
            for k, v in other_items.items():
                checkpoint[k] = v
        torch.save(checkpoint, fname)

    def load(self, fname: str):
        logging.info(f"Loading EMA model file: {fname}")
        fix_old_checkpoints.mimic_old_modules()
        checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
        new_state_dict = self.module.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key in new_state_dict
        }
        self.module.load_state_dict(filtered_state_dict, strict=False)
        del checkpoint
