# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import glob
import logging
import os
import random
from dataclasses import dataclass, field

import img2dataset
import pandas as pd
import simple_parsing


@dataclass
class DownloadConfig:
    cc12m_index: str = field(default="tests/test_files/c12m_10samples.tsv")
    cc12m_local_dir: str = field(default="cc12m/")
    validation_percentage: float = 0.2
    split_seed: int = 4
    skip_download: bool = False


def get_parser():
    parser = simple_parsing.ArgumentParser(
        description="pre-loading architecture", add_config_path_arg=True
    )
    parser.add_arguments(DownloadConfig, dest="options")
    return parser


def download(config: DownloadConfig) -> None:
    for d in [config.cc12m_local_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    if not config.skip_download:
        # download
        img2dataset.download(
            processes_count=16,
            thread_count=32,
            url_list=config.cc12m_index,
            resize_mode="no",
            input_format="tsv",
            output_folder=config.cc12m_local_dir,
            output_format="webdataset",
            url_col="url",
            caption_col="caption",
            number_sample_per_shard=1000,
            distributor="multiprocessing",
        )
    else:
        logging.info(f"Skipping cc12m download because --skip-download was passed")

    logging.info(f"Preparing TSVs")
    for pq_file in glob.glob(f"{config.cc12m_local_dir}/*.parquet"):
        bn = os.path.basename(pq_file)
        df = pd.read_parquet(pq_file, engine="pyarrow")
        df = df[df["status"] == "success"]
        out_df = pd.DataFrame(columns=["tar", "file", "caption"])
        out_df["file"] = df["key"] + ".jpg"
        out_df["caption"] = df[["caption"]]
        out_df["tar"] = pq_file.replace(".parquet", ".tar")
        output_path = f'{config.cc12m_local_dir}/{bn.replace(".parquet", ".tsv")}'
        out_df.to_csv(output_path, sep="\t", index=False)
        logging.info(f"wrote tsv to {output_path}")

    tsvs = [
        g for g in glob.glob(f"{config.cc12m_local_dir}/*.tsv") if "validation" not in g
    ]
    random.Random(config.split_seed).shuffle(tsvs)
    midpoint = int(len(tsvs) * config.validation_percentage)
    train_tsvs = tsvs[:midpoint]
    validation_tsvs = tsvs[midpoint:]

    # In the sample download case, just use the same tsv
    if len(tsvs) == 1:
        train_tsvs = tsvs
        validation_tsvs = tsvs

    # Write the list of training tsv files to the index
    with open("training_0.tsv", "w") as training_index_f:
        training_index_f.write("filename\n")
        training_index_f.write("\n".join(train_tsvs))
        training_index_f.write("\n")

    # Create a validation tsv
    with open(f"{config.cc12m_local_dir}/validation.tsv", "w") as validation_f:
        validation_f.write("tar\tfile\tcaption\t\n")

    for tsv in validation_tsvs:
        df = pd.read_csv(tsv, sep="\t")
        df.to_csv(
            f"{config.cc12m_local_dir}/validation.tsv",
            mode="a",
            index=False,
            header=False,
            sep="\t",
        )

    # Write the validation tsv to the validation index
    with open("validation.tsv", "w") as validation_index_f:
        validation_index_f.write("filename\n")
        validation_index_f.write(f"{config.cc12m_local_dir}/validation.tsv\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    download(args.options)
