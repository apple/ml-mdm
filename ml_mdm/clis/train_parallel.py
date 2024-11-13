# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
"""
eg command:
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 train_parallel.py
"""
import logging
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from ml_mdm import helpers, reader
from ml_mdm.language_models import factory

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torchinfo

import torch.distributed as dist
import torch.nn as nn

from ml_mdm import trainer
from ml_mdm.config import get_arguments, get_model, get_pipeline
from ml_mdm.distributed import init_distributed_singlenode
from ml_mdm.lr_scaler import LRScaler
from ml_mdm.models.model_ema import ModelEma
from ml_mdm.reader import convert
from ml_mdm.utils import simple_logger


def load_batch(next_sample, device):
    for key in next_sample:
        if next_sample[key].dtype.kind == "u" or next_sample[key].dtype.kind == "f":
            next_sample[key] = torch.from_numpy(next_sample[key]).to(
                dtype=torch.float32, device=device, non_blocking=True
            )

    if "watermark_score" in next_sample:
        next_sample["watermark_score"] = torch.tensor(
            [float(convert(w)) for w in next_sample["watermark_score"]]
        ).to(device=device, non_blocking=True)
    if "state" in next_sample:
        next_sample["scale"] = (
            float(next_sample["image"].size(2)) / next_sample["state"][:, 0]
        )
    return next_sample


def main(args):
    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)

    input_channels = 3
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    tokenizer, language_model = factory.create_lm(args, device=device)
    language_model_dim = language_model.embed_dim

    args.unet_config.conditioning_feature_dim = language_model_dim
    denoising_model = get_model(args.model)(
        input_channels, input_channels, args.unet_config
    ).to(device)
    torchinfo.summary(denoising_model)
    diffusion_model = get_pipeline(args.model)(
        denoising_model, args.diffusion_config
    ).to(device)

    if global_rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    if "MASTER_ADDR" in os.environ:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    other_items = None
    if (
        args.pretrained_vision_file is not None
        and args.pretrained_vision_file != ""
        and os.path.exists(args.pretrained_vision_file)
    ):
        logging.info(f"Loading ckpt from {args.pretrained_vision_file}")
        other_items = diffusion_model.model.vision_model.load(
            args.pretrained_vision_file
        )
        ema_model.load(args.pretrained_vision_file)

    if other_items is not None:
        start_batch_num = batch_num = other_items["batch_num"]
        exp_avg_loss = other_items["exp_avg_loss"]
        exp_avg_loss_var = other_items["exp_avg_loss_var"]
        best_avg_loss = other_items["best_avg_loss"]
        logging.info(f"Loaded model. Batch #: {batch_num}")
    else:
        exp_avg_loss = 0
        exp_avg_loss_var = 0
        best_avg_loss = 1e12
        start_batch_num = batch_num = 0

    logger = None
    if global_rank == 0:
        logger = simple_logger.Logger(
            os.path.join(args.output_dir, "train"), args.log_freq
        )
        logger.add_tensorboard_logger()

    if args.fp16:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    max_lr = args.lr
    # Should eps be 1e-4 like for LMs in fp16 ?
    if args.use_adamw:
        optimizer = torch.optim.AdamW(
            diffusion_model.model.vision_model.parameters(),
            lr=max_lr,
            weight_decay=0,
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.Adam(
            diffusion_model.model.vision_model.parameters(), lr=max_lr, eps=1e-8
        )
    lr_scaler = LRScaler()
    scheduler = lr_scaler.get_lr_scheduler(args.warmup_steps, optimizer)

    loss = nn.GaussianNLLLoss(reduction="mean")

    # tracking losses & contants
    total_loss_val = 0
    total_time = 0
    num_time_counts = 0
    counter = 0
    wt = 0.01
    CLIP = 3

    # intialize the model
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        model = nn.parallel.DistributedDataParallel(
            diffusion_model.model,
            device_ids=[local_rank],
        )
    else:
        model = diffusion_model.model
    diffusion_model.model = model
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    # Check if the model is wrapped in DistributedDataParallel
    ema_model = ModelEma(getattr(diffusion_model.model, "module", diffusion_model.model).vision_model)

    # get the dataloader
    if args.multinode:
        partition_id = 0
        num_partitions = 1
    else:
        partition_id = local_rank
        num_partitions = world_size
    train_loader = reader.get_dataset_partition(
        partition_id,
        num_partitions,
        tokenizer,
        args.batch_size,
        args.file_list,
        args.reader_config,
        args.num_epochs,
        load_numpy=args.use_precomputed_text_embeddings,
        is_index_file=True,
    )
    data_iter = iter(train_loader)
    next_sample = load_batch(data_iter.next(), device)

    while True:
        counter = (counter + 1) % args.num_gradient_accumulations
        batch_num += counter == 0
        accumulate_gradient = counter != 0
        if global_rank == 0:
            logger.batch_num = batch_num

        # load data
        sample = next_sample
        next_sample = load_batch(data_iter.next(), device)

        start_time = time.time()
        with torch.no_grad():
            images = (sample["image"].type(torch.float) - 127.0) / 128.0
            images = torch.permute(images, (0, 3, 1, 2))
            lm_outputs, lm_mask = language_model(sample, tokenizer)
            sample["lm_outputs"] = lm_outputs
            sample["lm_mask"] = lm_mask
            sample["images"] = images

        if accumulate_gradient:
            no_sync_context = diffusion_model.model.no_sync() if hasattr(diffusion_model.model, "no_sync") else nullcontext()
            with no_sync_context:
                loss_val, losses, times, x_t, means, targets = trainer.train_batch(
                    diffusion_model,
                    sample,
                    optimizer,
                    scheduler,
                    logger,
                    args,
                    grad_scaler=grad_scaler,
                    accumulate_gradient=accumulate_gradient,
                    num_grad_accumulations=args.num_gradient_accumulations,
                    ema_model=ema_model,
                    loss_factor=args.loss_factor,
                )
        else:
            loss_val, losses, times, x_t, means, targets = trainer.train_batch(
                diffusion_model,
                sample,
                optimizer,
                scheduler,
                logger,
                args,
                grad_scaler=grad_scaler,
                accumulate_gradient=accumulate_gradient,
                num_grad_accumulations=args.num_gradient_accumulations,
                ema_model=ema_model,
                loss_factor=args.loss_factor,
            )

        total_time += time.time() - start_time
        num_time_counts += 1
        if np.isnan(loss_val):
            continue
        # accumulate loss
        if batch_num != 1:
            # E[(x-E[x])^2] = E[x^2] - E[x]^2
            std_loss = np.sqrt(max(1, exp_avg_loss_var))
            delta_loss = loss_val - exp_avg_loss
            clipped_loss = exp_avg_loss + std_loss * CLIP * np.tanh(
                delta_loss / std_loss / CLIP
            )
            exp_avg_loss = exp_avg_loss * (1.0 - wt) + wt * clipped_loss
            exp_avg_loss_var = (
                exp_avg_loss_var * (1.0 - wt) + wt * (clipped_loss - exp_avg_loss) ** 2
            )
        else:
            std_loss = loss_val
            best_avg_loss = loss_val
            exp_avg_loss = loss_val
            exp_avg_loss_var = loss_val**2
        total_loss_val += loss_val
        # print(f"Allocated memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB", end='')
        # print(f"Val loss: {loss_val}")

        if (not accumulate_gradient) and (global_rank == 0):
            metrics = {
                "loss": loss_val,
                "batch_num": batch_num,
                "exp_avg_loss": exp_avg_loss,
                "step time": total_time / num_time_counts,
                "batch time": total_time / (batch_num - start_batch_num),
                "exp_avg_std_loss": np.sqrt(exp_avg_loss_var),
            }
            for k, v in metrics.items():
                logger.add_scalar(k, v)

            if batch_num % args.log_freq == 0:
                logging.info(f"Batch: {batch_num} - {metrics}")

            if (batch_num % args.save_freq == 0) or (
                batch_num == args.num_training_steps
            ):
                logging.info(f"Saving model. Batch = {batch_num}")
                vision_model_file = os.path.join(
                    args.output_dir, f"vis_model_{batch_num:06d}.pth"
                )
                vision_model_noema_file = os.path.join(
                    args.output_dir, f"vis_model_noema_{batch_num:06d}.pth"
                )
                other_items = {
                    "batch_num": batch_num,
                    "loss": loss_val,
                    "best_avg_loss": exp_avg_loss,
                    "exp_avg_loss": exp_avg_loss,
                    "exp_avg_loss_var": exp_avg_loss_var,
                    "args": args,
                }  # save full config.
                ema_model.save(vision_model_file, other_items=other_items)
                getattr(diffusion_model.model, "module", diffusion_model.model).vision_model.save(
                    vision_model_noema_file, other_items=other_items
                )
                torch.cuda.empty_cache()
                torch.mps.empty_cache()

        if (batch_num % args.save_freq == 0) or (batch_num == args.num_training_steps):
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if batch_num == args.num_training_steps:
            break

    # seems that GC collect might be causing problems if this is not set to none.
    train_loader = None


if __name__ == "__main__":
    args = get_arguments(mode="trainer")
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=getattr(logging, args.loglevel.upper(), None),
    )
    seed = args.seed
    if args.seed == -1:
        seed = int(time.time() % 10000)
    logging.info(f"Using seed: {seed}")
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.mps.empty_cache()
    helpers.print_args(args)
    main(args)
