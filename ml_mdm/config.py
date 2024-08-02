# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import simple_parsing
from simple_parsing.wrappers.field_wrapper import ArgumentGenerationMode

from ml_mdm import reader

MODEL_CONFIG_REGISTRY = {}
MODEL_REGISTRY = {}
PIPELINE_CONFIG_REGISTRY = {}
PIPELINE_REGISTRY = {}


def register_model_config(*names):
    arch, main = names

    def register_config_cls(cls):
        MODEL_CONFIG_REGISTRY[arch] = {}
        MODEL_CONFIG_REGISTRY[arch]["model"] = main
        MODEL_CONFIG_REGISTRY[arch]["config"] = cls
        return cls

    return register_config_cls


def register_model(*names):
    def register_model_cls(cls):
        for name in names:
            MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_pipeline_config(*names):
    def register_pipeline_cls(cls):
        for name in names:
            PIPELINE_CONFIG_REGISTRY[name] = cls
        return cls

    return register_pipeline_cls


def register_pipeline(*names):
    def register_pipeline_cls(cls):
        for name in names:
            PIPELINE_REGISTRY[name] = cls
        return cls

    return register_pipeline_cls


def get_model(name):
    if name not in MODEL_CONFIG_REGISTRY:
        raise NotImplementedError
    return MODEL_REGISTRY[MODEL_CONFIG_REGISTRY[name]["model"]]


def get_pipeline(name):
    if name not in MODEL_CONFIG_REGISTRY:
        raise NotImplementedError
    return PIPELINE_REGISTRY[MODEL_CONFIG_REGISTRY[name]["model"]]


def add_common_arguments(parser):
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level")
    parser.add_argument("--device", type=str, default="cuda", help="Logging frequency")
    parser.add_argument(
        "--fp16", type=int, default=0, help="Using fp16 to speed-up training"
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random number seed")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")

    # common language configs
    parser.add_argument(
        "--vocab_file", type=str, default="data/c4_wpm.vocab", help="WPM model file"
    )
    parser.add_argument(
        "--pretrained-vision-file",
        type=str,
        default=None,
        help="Choose either ema or non-ema file to start from",
    )
    parser.add_argument("--categorical-conditioning", type=int, default=0)
    parser.add_argument(
        "--text-model",
        type=str,
        default="google/flan-t5-xl",
        help="text model for encoding the caption",
    )
    parser.add_argument(
        "--model",
        "--vision-model",
        type=str,
        choices=list(MODEL_CONFIG_REGISTRY.keys()),
        default="unet",
    )  # currently, only one option
    # pre-computing text embeddings (we use it in trainer only)
    parser.add_argument(
        "--use-precomputed-text-embeddings",
        type=int,
        default=0,
        help="use precomputed text embeddings for conditioning.",
    )
    # batch information
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size to use")
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=850000,
        help="# of training steps to train for",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=20000, help="# of epochs to train for"
    )

    return parser


def get_trainer_parser(config_path=None):
    parser = simple_parsing.ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        add_config_path_arg=True,
        config_path=config_path,
        description="Multi-GPU training of diffusion models",
    )

    parser.add_argument(
        "--multinode",
        type=int,
        default=1,
        help="Whether to use multi node training",
    )
    parser.add_argument("--local-rank", type=int, default=0, help="for debugging")
    parser.add_argument("--use-adamw", action="store_true")
    parser.add_argument(
        "--file-list",
        type=str,
        default="cifar10-32/train.csv",
        help="List of training files in dataset. "
        "in moltinode model, this list is different per device,"
        "otherwise the list is shared with all devices in current node",
    )
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save-freq", type=int, default=1000, help="Saving frequency")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr-scaling-factor",
        type=float,
        default=0.8,
        help="Factor to reduce maximum learning rate",
    )
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=2.0, help="Gradient Clip Norm"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=5000, help="# of warmup steps"
    )
    parser.add_argument(
        "--num-gradient-accumulations",
        type=int,
        default=1,
        help="# of steps to accumulate gradients",
    )
    parser.add_argument(
        "--loss-factor",
        type=float,
        default=1,
        help="multiply the loss by a factor, to simulate old behaviors.",
    )
    parser.add_argument(
        "--resume-from-ema",
        action="store_true",
        help="If enabled, by default loading ema checkpoint when resume.",
    )

    return parser


def get_sampler_parser(config_path=None):
    parser = simple_parsing.ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        add_config_path_arg=True,
        config_path=config_path,
        description="Generate samples from diffusion model for evaluation",
    )

    parser.add_argument(
        "--model-file", type=str, default="", help="Path to saved model"
    )
    parser.add_argument(
        "--test-file-list", type=str, default="", help="List of test files in dataset"
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="samples",
        help="directory to keep all samples",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=1000, help="Minimum Evaluation interval"
    )
    parser.add_argument(
        "--sample-image-size", type=int, default=-1, help="Size of image"
    )
    parser.add_argument("--port", type=int, default=19231)
    parser.add_argument(
        "--min-examples",
        type=int,
        default=10000,
        help="minimum number of examples to generate",
    )
    return parser


def get_evaluator_parser(config_path=None):
    parser = simple_parsing.ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        add_config_path_arg=True,
        config_path=config_path,
        description="Simple diffusion model",
    )

    parser.add_argument(
        "--test-file-list", type=str, default="", help="List of test files in dataset"
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="samples",
        help="directory to keep all samples",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=1000, help="Minimum Evaluation interval"
    )
    parser.add_argument(
        "--sample-image-size", type=int, default=-1, help="Size of image"
    )
    parser.add_argument(
        "--num-eval-batches", type=int, default=500, help="# of batches to evaluate on"
    )
    return parser


def get_demo_parser(config_path=None):
    parser = simple_parsing.ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        add_config_path_arg=True,
        config_path=config_path,
        description="Generate samples from diffusion model for visualization",
    )

    parser.add_argument(
        "--sample-dir",
        type=str,
        default="samples",
        help="directory to keep all samples",
    )
    parser.add_argument(
        "--sample-image-size", type=int, default=-1, help="Size of image"
    )
    return parser


def get_preload_parser():
    parser = simple_parsing.ArgumentParser(description="pre-loading architecture")
    parser.add_argument(
        "--model",
        "--vision-model",
        type=str,
        choices=list(MODEL_CONFIG_REGISTRY.keys()),
        default="unet",
    )  # currently, only one option
    parser.add_argument(
        "--reader-config-file", type=str, default=None, help="Config file for reader"
    )
    parser.add_argument(
        "--model-config-file", type=str, default=None, help="Config file for model"
    )
    return parser


def get_arguments(args=None, mode="trainer", additional_config_paths=[]):
    from ml_mdm import diffusion, models

    pre_args, _ = get_preload_parser().parse_known_args(args)
    model_name = pre_args.model
    config_path = additional_config_paths
    if pre_args.reader_config_file is not None:
        config_path.append(pre_args.reader_config_file)
    if pre_args.model_config_file is not None:
        config_path.append(pre_args.model_config_file)
    if mode == "trainer":
        parser = get_trainer_parser(config_path)
    elif mode == "sampler":
        parser = get_sampler_parser(config_path)
    elif mode == "evaluator":
        parser = get_evaluator_parser(config_path)
    elif mode == "demo":
        parser = get_demo_parser(config_path)
    else:
        raise NotImplementedError

    # add common args
    parser = add_common_arguments(parser)

    # add submodule args
    parser.add_arguments(reader.ReaderConfig, dest="reader_config")

    # vision model configs
    parser.add_arguments(
        MODEL_CONFIG_REGISTRY[model_name]["config"], dest="unet_config"
    )
    parser.add_arguments(
        PIPELINE_CONFIG_REGISTRY[MODEL_CONFIG_REGISTRY[model_name]["model"]],
        dest="diffusion_config",
    )

    parser.confilct_resolver_max_attempts = 5000

    # parse known args
    args, _ = parser.parse_known_args(args)
    return args
