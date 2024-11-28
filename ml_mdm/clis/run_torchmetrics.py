# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import simple_parsing
import json
import logging
import os
import time

import PIL.Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

import numpy as np
import torch

from ml_mdm import helpers
from dataclasses import dataclass, field

@dataclass
class MetricsConfig:
    loglevel: str = field(default="INFO", 
        metadata={"help": "Logging level"})
    sample_dir: str = field(default="", 
        metadata={"help": "directory with samples"})
    metrics: str = field(default="clip,fid", 
        metadata={"help": "Metrics to compute(comma separated)"})
    reference_dir: str = field(default="", 
        metadata={"help": "directory with reference images"})
    num_samplers: int = field(default=1, 
        metadata={"help": "Number of jobs generating samples"})
    num_training_steps: int = field(default=850000, 
        metadata={"help": "# of training steps to train for"})
    max_caption_length: int = field(default=77, 
        metadata={"help": "Maximum length of caption"})
    eval_freq: int = field(default=1000, 
        metadata={"help": "Minimum Evaluation interval"})
    clip_model: str = field(default="openai/clip-vit-base-patch16", 
        metadata={"help": "Model to use for clip scores"})
    inception_layer_fid: int = field(default=2048, 
        metadata={
            "choices": [64, 192, 768, 2048], 
            "help": "Which layer of inception to use for fid"
        })
    
def get_parser():
    parser = simple_parsing.ArgumentParser(description="Compute metrics on samples from diffusion model")
    parser.add_arguments(MetricsConfig, dest="options")
    return parser

def load_captions_and_images(dir_name, args, override_path=None):
    map_files = []
    for i in range(args.num_samplers):
        map_file = os.path.join(dir_name, f"rank{i}", "lst.json")
        while not os.path.exists(map_file):
            # first one takes a while
            logging.info(f"Map file {map_file} does not exist")
            time.sleep(300)
        map_files.append(map_file)

    captions = []
    images = []

    for rank in range(args.num_samplers):
        with open(map_files[rank]) as f:
            file_contents = f.read()
        lst_maps = json.loads(file_contents)
        num_images = len(lst_maps)
        for image_num in range(num_images):
            caption = lst_maps[image_num][1]
            if not caption.isascii():
                continue
            if len(caption) > args.max_caption_length:
                caption = caption[: args.max_caption_length]
            captions.append(caption)
            image_path = lst_maps[image_num][0]
            if override_path is not None:
                image_file = "/".join(image_path.split("/")[-3:])
                image_path = f"{override_path}/{image_file}"
            images.append(np.asarray(PIL.Image.open(image_path)))

    return captions, images


def main(args):
    batch_size = 128
    device = torch.device("cuda", 0)
    helpers.print_args(args)
    reference_captions, reference_images = load_captions_and_images(
        args.reference_dir, args
    )

    metrics = args.metrics.split(",")
    compute_fid, compute_clip = False, False
    if "fid" in metrics:
        compute_fid = True
    if "clip" in metrics:
        compute_clip = True

    if compute_fid:
        fid = FrechetInceptionDistance(feature=args.inception_layer_fid).to(device)
        num_examples = len(reference_images)

        ref1 = reference_images[::2]
        for i in range(0, len(ref1), batch_size):
            end_index = min(len(ref1), i + batch_size)
            images = torch.tensor(np.stack(ref1[i:end_index])).to(device)
            images = images.permute(0, 3, 1, 2)
            fid.update(images, real=True)

        ref2 = reference_images[1::2]
        for i in range(0, len(ref2), batch_size):
            end_index = min(len(ref2), i + batch_size)
            images = torch.tensor(np.stack(ref2[i:end_index])).to(device)
            images = images.permute(0, 3, 1, 2)
            fid.update(images, real=False)
        try:
            fid_score = fid.compute().item()
            logging.info(f"FID - references- {fid_score}")
        except Exception as e:
            logging.error(f"Encountered error {e}")

    if compute_clip:
        clip = CLIPScore(model_name_or_path=args.clip_model).to(device)
        num_examples = len(reference_images)
        for i in range(0, num_examples, batch_size):
            images = torch.tensor(
                np.stack(reference_images[i : min(i + batch_size, num_examples)])
            ).to(device)
            images = images.permute(0, 3, 1, 2)
            captions = reference_captions[i : min(i + batch_size, num_examples)]
            clip.update(images, captions)

        clip_score = clip.compute().item()
        logging.info(f"CLIP - references = {clip_score}")

    # load sampled images.
    sample_captions, sample_images = load_captions_and_images(args.sample_dir, args)

    if compute_fid:
        fid = FrechetInceptionDistance(feature=args.inception_layer_fid).to(device)
        num_examples = len(reference_images)
        for i in range(0, num_examples, batch_size):
            images = np.stack(reference_images[i : min(i + batch_size, num_examples)])
            images = torch.tensor(images).to(device).permute(0, 3, 1, 2)
            fid.update(images, real=True)

        num_examples = len(sample_images)
        for i in range(0, num_examples, batch_size):
            images = torch.tensor(
                np.stack(sample_images[i : min(i + batch_size, num_examples)])
            ).to(device)
            images = images.permute(0, 3, 1, 2)
            fid.update(images, real=False)

        fid_score = fid.compute().item()
        logging.info(f"FID = {fid_score}")

    if compute_clip:
        clip = CLIPScore(model_name_or_path=args.clip_model).to(device)
        num_examples = len(sample_images)
        for i in range(0, num_examples, batch_size):
            images = torch.tensor(
                np.stack(sample_images[i : min(i + batch_size, num_examples)])
            ).to(device)
            captions = sample_captions[i : min(i + batch_size, num_examples)]
            images = images.clone().detach().permute(0, 3, 1, 2)
            clip.update(images, captions)

        clip_score = clip.compute().item()
        logging.info(f"Clip Score = {fid_score}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.options.loglevel.upper(), None),
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    helpers.print_args(args.options)
    main(args.options)
