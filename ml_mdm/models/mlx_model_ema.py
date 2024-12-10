import mlx.core as mx
import mlx.nn as nn

from ml_mdm.utils import fix_old_checkpoints

from copy import deepcopy
import logging
import torch

class ModelEmaMLX(nn.Module):
    def __init__(self, model, decay=0.9999, warmup_steps=0, device=None):
        super(ModelEmaMLX, self).__init__()
        
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device

        if self.device is not None:
            self.module.to(device=device)

        self.warmup_steps = warmup_steps
        self.counter = 0    

    def update(self, model):
        decay = (self.counter >= self.warmup_steps) * self.decay    
        self.counter += 1
        with mx.stop_gradient():
            msd = model.state_dict()
            for k, ema_v in self.module.state_dict().items():
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.mul_(decay).add_(model_v, alpha=(1.0 - decay))

    def save(self, fname, other_items=None):
        logging.info(f"Saving EMA model file: {fname}")
        checkpoint = {"state_dict": self.module.state_dict()}
        if other_items is not None:
            for k, v in other_items.items():
                checkpoint[k] = v
        mx.save(fname, checkpoint)

    def load(self, fname):
        logging.info(f"Loading EMA model file: {fname}")
        fix_old_checkpoints.mimic_old_modules()
        checkpoint = mx.load(fname)
        new_state_dict = self.module.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key in new_state_dict
        }
        self.module.load_state_dict(filtered_state_dict, strict=False)
        del checkpoint
