# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
from argparse import Namespace

from ml_mdm import config, reader
from ml_mdm.clis import generate_batch
from ml_mdm.language_models import factory


def test_small_batch():
    args = Namespace(
        batch_size=10,
        test_file_list="tests/test_files/sample_training_0.tsv",
        reader_config=reader.ReaderConfig(num_readers=1, reader_buffer_size=10),
        cfg_weight=1.1,  # test for negative prompts
        min_examples=2,
        vocab_file="data/t5.vocab",
        text_model="google/flan-t5-small",
        categorical_conditioning=0,
        use_precomputed_text_embeddings=0,
        fp16=0,
    )
    tokenizer, language_model = factory.create_lm(args=args, device="cpu")
    samples, num_samples = generate_batch.generate_data(
        device="cpu",
        local_rank=0,
        world_size=1,
        tokenizer=tokenizer,
        language_model=language_model,
        args=args,
    )
    assert num_samples > 0 and samples


def test_generate_batch():
    args = config.get_arguments(mode="sampler")
    args.batch_size = 10
    args.test_file_list = "tests/test_files/sample_training_0.tsv"
    args.reader_config = reader.ReaderConfig(num_readers=1, reader_buffer_size=10)
    args.min_examples = 2
    args.vocab_file = "data/t5.vocab"
    args.text_model = "google/flan-t5-small"
    pass
