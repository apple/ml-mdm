# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
from dataclasses import dataclass, field

from dataclass_wizard import YAMLWizard

import mlx.data as dx
import mlx.data.core
import numpy as np
from mlx.data import Buffer, Stream

from ml_mdm.language_models.tokenizer import Tokenizer


@dataclass
class ReaderConfig(YAMLWizard):
    smaller_side_size: int = field(
        default=-1, metadata={"help": "Smaller side is resized to this value"}
    )
    max_caption_length: int = field(
        default=-1, metadata={"help": "Maximum length of captions"}
    )
    max_token_length: int = field(
        default=-1,
        metadata={"help": "Maximum length of after tokenization of captions"},
    )
    image_size: int = field(default=-1, metadata={"help": "Size of image to resize to"})
    random_crop: bool = field(
        default=False,
        metadata={"help": "if true, using random crop instead of center crop"},
    )
    num_kept_files: int = field(
        default=-1,
        # note that num_kept_files should be tinkered with carefully
        # since some behaviors can have exotic effects.
        metadata={"help": "Maximum number of files to keep in mlx.data"},
    )
    num_readers: int = field(default=16, metadata={"help": "Number of working threads"})
    shuffle_buffer_size: int = (
        field(  # TODO (jack_carlson): is this field used anywhere
            default=9600, metadata={"help": "# of elements to buffer for shuffling"}
        )
    )
    reader_buffer_size: int = field(
        default=9600, metadata={"help": "# of batches prefetched"}
    )
    endpoint_url: str = field(
        default="",
        metadata={"help": "Url of s3 endpoint"},
    )
    bucket: str = field(default="mlx", metadata={"help": "s3 Bucket in endpoint"})
    prepad_caption_with_space: bool = field(
        default=True,
        metadata={"help": "Prepad caption with space (or mlx tokenizes wrongly)"},
    )
    use_tokenizer_scores: bool = field(
        default=True, metadata={"help": "Use scores in tokenization"}
    )
    prepad_bos: bool = field(
        default=False, metadata={"help": "Prepad tokenization with bos symbol"}
    )
    append_eos: bool = field(
        default=True, metadata={"help": "Append tokenization with eos symbol"}
    )
    padding_token: str = field(
        default="<pad>", metadata={"help": "Padding token to use"}
    )
    pad_to_max_length: bool = field(
        default=False, metadata={"help": "Pad the input text to maximum length"}
    )

    @classmethod
    def from_file(cls, config_file: str):
        with open(config_file, "r") as f:
            config_str = f.read()

        return cls.from_yaml(config_str)

    def save(self, config_file: str):
        self.to_yaml_file(config_file)


def _get_dataset(
    dataset: Stream,
    tokenizer: Tokenizer,
    batch_size: int,
    file_list: str,  # TODO (jack_carlson): this is never used
    config: ReaderConfig,
    skip_images: bool = False,
    load_numpy: bool = False,
) -> Stream:
    """
    Augment an existing dataset
    """
    # return dataset
    if not skip_images:
        dataset = dataset.read_from_tar("tar", "file", "image", from_key=True)
        dataset = dataset.load_image("image", "", format="RGB", from_memory=True)

        if config.image_size != -1:
            dataset = dataset.image_resize_smallest_side("image", config.image_size)
            dataset = dataset.image_center_crop(
                "image", config.image_size, config.image_size
            )

    if load_numpy:
        # NOTE: untested for latest mlx version.
        dataset = dataset.read_from_tar(
            "text_tar", "text_file", "text_embedding", from_key=True
        )
        dataset = dataset.load_numpy("text_embedding", "", from_memory=True)

    if tokenizer is not None:
        dataset = dataset.pad("caption", 0, 1, 0, ord(" "))
        # to fix mlx-data bug for strings with tokens that are unknown
        dataset = dataset.pad("caption", 0, 0, 1, ord(" "))

        if config.use_tokenizer_scores:
            dataset = dataset.tokenize(
                "caption",
                tokenizer.trie,
                ignore_unk=True,
                trie_key_scores=tokenizer.trie_key_scores,
                output_key="tokens",
            )

        # TODO: remove the prepadding options.
        if config.prepad_bos:
            dataset = dataset.pad("tokens", 0, 1, 0, tokenizer.bos)
        if config.append_eos:
            dataset = dataset.pad("tokens", 0, 0, 1, tokenizer.eos)
        if config.max_caption_length != -1:
            dataset = dataset.filter_by_shape(
                "caption", 0, 0, config.max_caption_length
            )

        if config.max_token_length != -1:
            dataset = dataset.filter_by_shape("tokens", 0, 0, config.max_token_length)
            if config.pad_to_max_length:
                pad_token = tokenizer.token_id(config.padding_token)
                dataset = dataset.pad_to_size(
                    "tokens", 0, config.max_token_length, pad_token
                )

    if tokenizer is not None:
        # pad with padding token not eos token
        pad_token = tokenizer.token_id(config.padding_token)
        dataset = dataset.batch(batch_size, pad={"tokens": pad_token})
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(config.reader_buffer_size, config.num_readers)
    return dataset


def get_dataset(
    tokenizer,
    batch_size,
    file_list: str,
    config: ReaderConfig,
    num_epochs: int = -1,
    skip_images: bool = False,
    load_numpy: bool = False,
    is_index_file: bool = False,
) -> Stream:
    dataset = dx.stream_csv_reader(file_list, "\t")
    dataset = dataset.repeat(num_epochs)
    if is_index_file:
        dataset = dataset.csv_reader_from_key("filename", "\t", quote='"')
    return _get_dataset(
        dataset, tokenizer, batch_size, file_list, config, skip_images, load_numpy
    )


def get_dataset_partition(
    partition_num,
    num_partitions,
    tokenizer,
    batch_size,
    file_list: str,
    config: ReaderConfig,
    num_epochs: int = -1,
    skip_images: bool = False,
    load_numpy: bool = False,
    is_index_file: bool = False,
):
    dataset = dx.stream_csv_reader(file_list, "\t")
    dataset = dataset.repeat(num_epochs)
    if is_index_file:
        dataset = dataset.csv_reader_from_key("filename", "\t", quote='"')
    if num_partitions != 1:
        dataset = dataset.partition(num_partitions, partition_num)
    return _get_dataset(
        dataset, tokenizer, batch_size, file_list, config, skip_images, load_numpy
    )


def convert(arr):
    arr = arr.astype(np.uint8)
    arr = arr[arr != 0]
    return "".join([chr(x) for x in arr])


def process_text(
    text: list[str], tokenizer: Tokenizer, config: ReaderConfig
) -> list[list[int]]:
    if config.use_tokenizer_scores:
        mlx_tokenizer = mlx.data.core.Tokenizer(
            tokenizer._trie, ignore_unk=True, trie_key_scores=tokenizer.trie_key_scores
        )
    else:
        mlx_tokenizer = mlx.data.core.Tokenizer(tokenizer._trie, ignore_unk=True)
    padded_tokens = []
    max_len = 0
    for d in text:
        if config.max_caption_length > -1:
            d = d[: config.max_caption_length]
        if config.prepad_caption_with_space:
            d = " " + d

        tokens = mlx_tokenizer.tokenize_shortest(d)
        if config.prepad_bos:
            tokens = [tokenizer.bos] + tokens
        if config.append_eos:
            tokens = tokens + [tokenizer.eos]
        if len(tokens) > max_len:
            max_len = len(tokens)
        if len(tokens) < config.max_token_length:
            pad_length = config.max_token_length - len(tokens)
            tokens = tokens + [tokenizer.token_id(config.padding_token)] * pad_length
        padded_tokens.append(tokens)
    if config.pad_to_max_length:
        max_len = config.max_token_length
    else:
        max_len = min(max_len, config.max_token_length)
    padded_tokens = [tokens[:max_len] for tokens in padded_tokens]
    return padded_tokens


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser(description="Reader")
    parser.add_arguments(ReaderConfig, dest="reader_config")
    parser.add_argument(
        "--vocab_file", type=str, default="data/c4_wpm.vocab", help="WPM model file"
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default="training.tsv",
        help="List of training files in dataset",
    )
    # Note that it is called add_arguments, not add_argument.
    args = parser.parse_args()
    logging.basicConfig(
        level="INFO",
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    tokenizer = Tokenizer(args.vocab_file)
    loader = get_dataset(
        tokenizer,
        2,
        args.file_list,
        args.reader_config,
        num_epochs=1,
        is_index_file=True,
    )
    for sample in loader:
        images = sample["image"].transpose(0, 3, 1, 2)
        print(images.shape)
        break
