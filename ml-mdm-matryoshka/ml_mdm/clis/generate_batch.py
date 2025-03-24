# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import glob
import json
import logging
import os
import time

import PIL.Image
import tqdm
from einops import repeat

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from ml_mdm import generate_html, helpers, reader
from ml_mdm.clis.train_parallel import load_batch  # TODO this doesnt exist!
from ml_mdm.config import get_arguments, get_model, get_pipeline
from ml_mdm.distributed import init_distributed_singlenode
from ml_mdm.language_models import factory
from ml_mdm.reader import convert


def generate_data(device, local_rank, world_size, tokenizer, language_model, args):
    """Create the (image, text) pairs that will be used for all evaluations."""

    def _create_loader():
        # read the data. num_epochs is not 1 because the input file can
        # have fewer examples than what we want to genereate for evaluation
        # (e.g. cifar10 validation set is 10K, but we might want to generate
        # 50K samples for Fid computation.
        return reader.get_dataset_partition(
            local_rank,
            world_size,
            tokenizer,
            args.batch_size,
            args.test_file_list,
            args.reader_config,
            num_epochs=1000,
            skip_images=False,
            is_index_file=True,
        )

    samples = []
    num_samples = 0
    negative_tokens = np.asarray(
        reader.process_text(["low quality"], tokenizer, args.reader_config)
    )

    for sample in _create_loader():
        with torch.no_grad():
            sample = load_batch(sample, device)
            if getattr(args, "cfg_weight", 1) > 1:  # include negative prompts
                batch_size = sample["tokens"].shape[0]
                neg_tokens = repeat(negative_tokens, "() c -> b c", b=batch_size)
                len_max = max(sample["tokens"].shape[1], neg_tokens.shape[1])
                new_tokens = np.zeros((batch_size * 2, len_max)).astype(
                    neg_tokens.dtype
                )
                new_tokens[:batch_size, : neg_tokens.shape[1]] = neg_tokens
                new_tokens[batch_size:, : sample["tokens"].shape[1]] = sample["tokens"]
                sample["tokens"] = new_tokens
                for key in ["scale", "watermarker_score"]:
                    if key in sample:
                        sample[key] = torch.cat([sample[key], sample[key]], 0)
            lm_outputs, lm_mask = language_model(sample, tokenizer)
            num_samples += sample["image"].shape[0]
            sample["lm_outputs"] = lm_outputs
            sample["lm_mask"] = lm_mask
            for key in sample:
                if isinstance(sample[key], torch.Tensor) and sample[key].is_cuda:
                    sample[key] = sample[key].cpu()
            samples.append(sample)
            if num_samples * world_size >= args.min_examples:
                break
    return samples, num_samples


def main(args):
    helpers.print_args(args)
    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=1000)
    if getattr(args, "global_world_size", None) is not None:
        world_size = args.global_world_size
        global_rank = 8 * args.global_offset + local_rank

    print(
        f"Local rank: {local_rank}, global_rank={global_rank}."
        + f"World size={world_size}"
    )
    device = torch.device("cuda")
    input_channels = 3
    tokenizer, language_model = factory.create_lm(args, device=device)
    language_model_dim = language_model.embed_dim

    args.unet_config.conditioning_feature_dim = language_model_dim
    denoising_model = get_model(args.model)(
        input_channels, input_channels, args.unet_config
    ).cuda()
    diffusion_model = get_pipeline(args.model)(
        denoising_model, args.diffusion_config
    ).to(device)
    model = nn.parallel.DistributedDataParallel(
        diffusion_model.model, device_ids=[local_rank]
    )
    diffusion_model.model = model
    if local_rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    eval_data, num_examples = generate_data(
        device, local_rank, world_size, tokenizer, language_model, args
    )
    if num_examples * world_size < args.min_examples:
        logging.fatal(
            f"Number of examples read (={num_examples})"
            + f" was less than needed (={args.min_examples})"
        )

    # save images and captions to reference folder for downstream evals.
    reference_dir = os.path.join(args.sample_dir, "references", f"rank{local_rank}")

    os.makedirs(reference_dir, exist_ok=True)
    caption_lst = []
    num_examples_saved = 0
    for sample in eval_data:
        images_np = sample["image"].cpu().numpy().astype(np.uint8)
        for i, image_np in enumerate(images_np):
            dest_fn = os.path.join(
                reference_dir, f"sample_{num_examples_saved:06d}.png"
            )
            PIL.Image.fromarray(image_np, "RGB").save(dest_fn)
            caption = sample["caption"][i]
            caption = reader.convert(caption)
            caption_lst.append((dest_fn, caption))
            num_examples_saved += 1
            if num_examples_saved * world_size >= args.min_examples:
                break

    # serialize map to json and store in shared nfs
    reference_file = os.path.join(reference_dir, "lst.json")
    logging.info(f"Writing mapping file to : {reference_file}")
    with open(reference_file, "w") as write:
        json.dump(caption_lst, write)

    if local_rank == 0:
        generate_html.create_html(
            os.path.join(args.sample_dir, "references", "index.html"), 64, caption_lst
        )

    # Time to start generating images.
    assert args.sample_image_size != -1
    last_batch = -100000000000
    best_err = np.Inf

    vision_model_file = args.model_file
    assert os.path.exists(vision_model_file)

    if getattr(args, "threshold_function", None) is not None:
        from samplers import ThresholdType

        diffusion_model.sampler._config.threshold_function = {
            "clip": ThresholdType.CLIP,
            "dynamic (Imagen)": ThresholdType.DYNAMIC,
            "dynamic (DeepFloyd)": ThresholdType.DYNAMIC_IF,
            "none": ThresholdType.NONE,
        }[args.threshold_function]

    gammas = diffusion_model.sampler.gammas.detach().cpu().numpy().copy()
    # Wait for all readers to arrive before loading model file.
    torch.distributed.barrier()
    logging.info(f"[{local_rank}] Loading file: {vision_model_file}")
    other_items = diffusion_model.model.module.load(vision_model_file)
    batch_num = other_items["batch_num"]
    torch.distributed.barrier()
    last_batch = batch_num
    logging.info(f"Generating samples. Step: {batch_num}")
    sample_dir = os.path.join(
        args.sample_dir, f"checkpoint_{batch_num}", f"rank{local_rank}"
    )

    logging.info(f"Creating directory {sample_dir}")
    os.makedirs(sample_dir, exist_ok=True)
    samples_file = os.path.join(sample_dir, "lst.json")

    sample_count = 0
    for sample in tqdm.tqdm(eval_data):
        with torch.no_grad():
            num_samples = sample["image"].shape[0]
            for key in sample:
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].to(device)

            samples = diffusion_model.sample(
                num_samples,
                sample,
                args.sample_image_size,
                device,
                return_sequence=False,
                resample_steps=hasattr(args, "num_inference_steps"),
                num_inference_steps=getattr(args, "num_inference_steps", 1000),
                ddim_eta=getattr(args, "ddim_eta", 1.0),
                guidance_scale=getattr(args, "cfg_weight", 1.0),
            )
            samples = torch.clamp(samples * 128.0 + 127.0, min=0, max=255)
            samples_np = samples.cpu().permute(0, 2, 3, 1).to(torch.uint8).numpy()
            for sample_np in samples_np:
                dest_fn = os.path.join(sample_dir, f"sample_{sample_count:06d}.png")
                PIL.Image.fromarray(sample_np, "RGB").save(dest_fn)
                # same caption, just different image
                caption_lst[sample_count] = (dest_fn, caption_lst[sample_count][1])
                sample_count += 1
                if sample_count * world_size >= args.min_examples:
                    logging.info(f"Writing mapping file to : {samples_file}")
                    if local_rank == 0:
                        generate_html.create_html(
                            os.path.join(
                                args.sample_dir, f"checkpoint_{batch_num}", "index.html"
                            ),
                            64,
                            caption_lst,
                        )
                    with open(samples_file, "w") as write:
                        json.dump(caption_lst, write)
                    break
            if sample_count * world_size >= args.min_examples:
                break


if __name__ == "__main__":
    args = get_arguments(mode="sampler")
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), None),
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    main(args)
