# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import argparse

import torch

from ml_mdm import reader
from ml_mdm.clis.train_parallel import load_batch
from ml_mdm.language_models import factory


def test_get_dataset():
    tokenizer = factory.create_tokenizer("data/t5.vocab")
    dataset = reader.get_dataset(
        tokenizer=tokenizer,
        batch_size=2,
        file_list="tests/test_files/sample_training_0.tsv",
        config=reader.ReaderConfig(
            num_readers=1,
            reader_buffer_size=10,
            image_size=40,
            use_tokenizer_scores=True,
        ),
        is_index_file=True,
    )
    sample = next(dataset)
    assert sample is not None
    assert "tokens" in sample
    assert "image" in sample
    assert sample["image"].shape == (2, 40, 40, 3)


def test_get_dataset_partition():
    tokenizer = factory.create_tokenizer("data/t5.vocab")
    train_loader = reader.get_dataset_partition(
        partition_num=0,
        num_partitions=1,
        tokenizer=tokenizer,
        batch_size=2,
        file_list="tests/test_files/sample_training_0.tsv",
        config=reader.ReaderConfig(num_readers=1, reader_buffer_size=10),
        is_index_file=True,
    )
    assert train_loader
    assert next(train_loader)


def test_process_text():
    line = "A bicycle on top of a boat."
    tokenizer = factory.create_tokenizer("data/t5.vocab")
    tokens = reader.process_text(
        [line], tokenizer=tokenizer, config=reader.ReaderConfig()
    )
    assert len(tokens) > 0
    assert len(tokens[0]) > 0


test_get_dataset()