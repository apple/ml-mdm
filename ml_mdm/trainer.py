# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
from argparse import Namespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def train_batch(
    model: torch.nn.Module,
    sample: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: Optional[torch.utils.tensorboard.SummaryWriter],
    args: Namespace,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulate_gradient: bool = False,
    num_grad_accumulations: int = 1,
    ema_model: Optional[nn.Module] = None,
    loss_factor: float = 1.0,
):
    breakpoint()
    model.train()
    lr = scheduler.get_last_lr()[0]
    # Updates the scale for next iteration
    if args.fp16:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            losses, times, x_t, means, targets, weights = model.get_loss(sample)
            if weights is None:
                loss = losses.mean()
            else:
                loss = (losses * weights).sum() / weights.sum()
            loss = loss * loss_factor  # TODO: to simulate old behaviors
            loss_val = loss.item()

            if np.isnan(loss_val):
                optimizer.zero_grad()
                return loss_val, losses, times, x_t, means, targets

            if num_grad_accumulations != 1:
                loss = loss / num_grad_accumulations
        # Unscales gradients and calls or skips optimizer.step()
        grad_scaler.scale(loss).backward()

        if not accumulate_gradient:
            # Unscales the gradients of optimizer's assigned params in-place
            grad_scaler.unscale_(optimizer)
            # model.module.rescale_gradient_norms(args.gradient_clip_norm)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.model.parameters(), args.gradient_clip_norm
            ).item()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if ema_model is not None:
                ema_model.update(
                    getattr(model.model, "module", model.model).vision_model
                )
    else:
        losses, times, x_t, means, targets, weights = model.get_loss(sample)
        if weights is None:
            loss = losses.mean()
        else:
            loss = (losses * weights).sum() / weights.sum()
        loss_val = loss.item()
        if np.isnan(loss_val):
            # total_loss.backward() # is backward needed
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
            return loss_val, losses, times, x_t, means, targets

        loss.backward()
        if num_grad_accumulations != 1:
            loss = loss / num_grad_accumulations
        if not accumulate_gradient:
            total_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            ).item()
            optimizer.step()
            if ema_model is not None:
                ema_model.update(
                    getattr(model.model, "module", model.model).vision_model
                )

    if logger is not None and not accumulate_gradient:
        logger.add_scalar("train/Loss", loss_val)
        logger.add_scalar("lr", lr)

    if not accumulate_gradient:
        optimizer.zero_grad()
        scheduler.step()

    return loss_val, losses, times, x_t, means, targets
