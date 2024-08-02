# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
"""Program to download training / eval data.

It takes in data config file with regular expressions for training and eval

train:
    files:
        - s3://mlx/dataset/imagenet-64px/
        - s3://mlx/dataset/CC12M-64px/training.tsv
eval:
    files:
        - s3://mlx/dataset/CC12M-64px/validation.tsv

In addtion it takes arguments for the subset that this job will handle
(training or eval or all), and what node number the job is and how many
nodes this data will be distributed over.
"""

import argparse
import csv
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import boto3.session
import yaml

import mlx.data

from ml_mdm import helpers, s3_helpers


def read_tsv(filename):
    # Open the TSV file for reading
    with open(filename, "r", newline="") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        return [row for row in reader]


def write_tsv(filename, data):
    # Open a TSV file for writing
    with open(filename, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        for row in data:
            writer.writerow(row)


def add_path_to_field(local_file, field="tar", parent_dir=None):
    if parent_dir is None:
        parent_dir = str(Path(local_file).parent)
        if parent_dir[-1] != "/":
            parent_dir += "/"

    csv_out = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    writer = csv.writer(
        csv_out, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    field_index = -1
    tar_files = {}
    num_exceptions = 0
    with open(local_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
        first = True
        while True:
            try:
                row = next(reader)
                if first:
                    for i, x in enumerate(row):
                        if x == field:
                            field_index = i
                            break
                    assert field_index != -1
                    writer.writerow(row)
                    first = False
                    continue
                if parent_dir not in row[field_index]:
                    tar_file = parent_dir + row[field_index].split("/")[-1]
                    row[field_index] = tar_file
                tar_file = row[field_index]
                if tar_file not in tar_files:
                    tar_files[tar_file] = 1

                writer.writerow(row)
            except csv.Error:
                num_exceptions += 1
            except StopIteration:
                break

    logging.info(f"Copying {csv_out.name} to {local_file}")
    if num_exceptions != 0:
        logging.warning(
            f"WARNING. {local_file} raised {num_exceptions}"
            + f"exceptions during reading. "
        )
    # now copy csv file back to local_file.
    shutil.copy(csv_out.name, local_file)
    return tar_files


def get_files(
    tsv_patterns,
    output_file,
    node_num,
    num_nodes,
    endpoint_url=s3_helpers.ENDPOINT_URL,
    download_tar=True,
    no_bandwidth=False,
    pretrained_text_embeddings=None,
):
    # BUCKET = "mlx"
    num_concurrent_fetches = 5
    logging.info(f"Get files. Node # {node_num} of {num_nodes}")
    # expand the regular expressions to actual files.
    files = []
    for tsv_pattern in tsv_patterns:
        cur_files = s3_helpers.get_file_list(tsv_pattern, endpoint_url=endpoint_url)
        if len(cur_files) == 0:
            raise Exception(f"No file found for regexp {tsv_pattern}")
        files.extend(cur_files)
    num_files = len(files)
    logging.info(f"Num files: {num_files}")
    remainder = num_files % num_nodes
    num_files_for_node = num_files // num_nodes
    if node_num < remainder:
        # firsrt <remainder> nodes need to handle an extra file each.
        start_index = (num_files_for_node + 1) * node_num
        end_index = start_index + num_files_for_node + 1
    else:
        start_index = num_files_for_node * node_num + remainder
        end_index = start_index + num_files_for_node
    remainder = num_files % num_nodes
    assert end_index - start_index > 0

    logging.info(
        f"Node # {node_num}. "
        + f"File Indices: {start_index}-{end_index} of {num_files}"
    )

    files = files[start_index:end_index]
    for i in range(len(files)):
        bucket_name, parent_path, pattern = s3_helpers._parse_path(files[i])
        files[i] = os.path.join(parent_path, pattern)
    logging.info(f"Downloading {len(files)} files.")
    ff = mlx.data.core.AWSFileFetcher(
        endpoint=endpoint_url,
        bucket=bucket_name,
        prefix="",
        local_prefix="",
        num_threads=16,
        num_prefetch_max=3,
        verbose=True,
    )
    num_files_cur = len(files)
    for i in range(0, len(files), num_concurrent_fetches):
        cur_files = files[i : min(i + num_concurrent_fetches, len(files))]
        ff.prefetch(cur_files)
        for fname in cur_files:
            ff.fetch(fname)

    if pretrained_text_embeddings is not None:
        logging.info(f"Downloading text embeddings")
        ff = mlx.data.AWSFileFetcher(
            endpoint=endpoint_url,
            bucket="jiatao-datasets/text2image",
            prefix="",
            local_prefix="datasets",
            num_threads=16,
            num_prefetch_max=3,
            verbose=True,
        )

        def _proc(x):
            dname, fname = x.split("/")[-2], x.split("/")[-1]
            return f"{dname}/{dname}_{pretrained_text_embeddings}/{fname}"

        files = [_proc(ff) for ff in files]
        for i in range(0, len(files), num_concurrent_fetches):
            cur_files = files[i : min(i + num_concurrent_fetches, len(files))]
            ff.prefetch(cur_files)
            for fname in cur_files:
                ff.fetch(fname)
        files = ["datasets/" + fi for fi in files]

    # note that, if using pretrained text emebddings, training config will be override
    index_file = open(output_file, "w")
    index_file.write("filename\n")
    for local_file in files:
        index_file.write(f"{local_file}\n")
    index_file.close()

    if no_bandwidth:
        max_tar_download_bandwidth = None
    else:
        max_tar_download_bandwidth = (1000 * 1000 * 1000) // num_nodes

    num_downloaded_tar, num_queued_tar = 0, 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        parent_dir = None if not pretrained_text_embeddings else ""
        futures = [
            executor.submit(add_path_to_field, key, parent_dir=parent_dir)
            for key in files
        ]
        download_futures = []
        for future in as_completed(futures):
            tar_files = future.result()
            if download_tar:
                # need to download files, since trainer will need them.
                for tar_file in tar_files:
                    download_futures.append(
                        executor.submit(
                            s3_helpers.download_object,
                            *[
                                bucket_name,
                                tar_file.replace(
                                    "_annoted", ""
                                ),  # HACK: FIXME in future
                                tar_file,
                                endpoint_url,
                                max_tar_download_bandwidth,
                            ],
                        )
                    )
                    num_queued_tar += 1
                    if num_queued_tar - num_downloaded_tar >= num_concurrent_fetches:
                        done, not_done = wait(
                            download_futures, return_when=FIRST_COMPLETED
                        )
                        for future in done:
                            logging.info(f"Downloaded {future.result()}")
                            num_downloaded_tar += 1
                            download_futures.remove(future)

        if download_tar:
            for future in as_completed(download_futures):
                logging.info(f"Downloaded {future.result()}")

        if pretrained_text_embeddings:
            num_downloaded_tar, num_queued_tar = 0, 0
            futures = [
                executor.submit(
                    add_path_to_field, key, field="text_tar", parent_dir=parent_dir
                )
                for key in files
            ]
            download_futures = []
            for future in as_completed(futures):
                tar_files = future.result()
                for tar_file in tar_files:
                    download_futures.append(
                        executor.submit(
                            s3_helpers.download_object,
                            *[
                                "jiatao-datasets",
                                "text2image/" + tar_file[9:],
                                tar_file,
                                endpoint_url,
                                max_tar_download_bandwidth,
                            ],
                        )
                    )
                    num_queued_tar += 1
                    if num_queued_tar - num_downloaded_tar >= num_concurrent_fetches:
                        done, not_done = wait(
                            download_futures, return_when=FIRST_COMPLETED
                        )
                        for future in done:
                            logging.info(f"Downloaded {future.result()}")
                            num_downloaded_tar += 1
                            download_futures.remove(future)

        logging.info(f"Finished job {node_num}")


def main(args):
    dataset_config_files = args.dataset_config_file.split(":")
    output_files = []
    for it, dataset_config_file in enumerate(dataset_config_files):
        with open(dataset_config_file, "r") as file:
            config = yaml.safe_load(file)

        if args.subset == "train":
            endpoint_url = config["train"].get("endpoint_url", args.endpoint_url)
            endpoint_url = endpoint_url if endpoint_url else s3_helpers.ENDPOINT_URL
            output_file = f"training_{args.worker_id}.tsv"
            if it > 0:
                output_file = output_file + f".{it}.tsv"
            get_files(
                config["train"]["files"],
                output_file,
                args.worker_id,
                args.num_downloaders,
                endpoint_url=endpoint_url,
                download_tar=args.download_tar,
                no_bandwidth=args.no_bandwidth,
                pretrained_text_embeddings=args.pretrained_text_embeddings,
            )
            output_files += [output_file]

        if args.subset == "eval":
            # only one downloader for eval. only the first dataset config is useful.
            endpoint_url = config["eval"].get("endpoint_url", args.endpoint_url)
            endpoint_url = endpoint_url if endpoint_url else s3_helpers.ENDPOINT_URL
            get_files(
                config["eval"]["files"],
                f"validation.tsv",
                0,
                1,
                endpoint_url=endpoint_url,
                download_tar=args.download_tar,
                no_bandwidth=args.no_bandwidth,
            )
            break

    if len(output_files) > 1:  # merge training set
        import random

        head, data = [], []
        for i, o in enumerate(output_files):
            d = read_tsv(o)
            if i == 0:
                head = [d[0]]
            data += d[1:]
        random.shuffle(data)
        data = head + data
        write_tsv(output_files[0], data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download tar files referred to in index file from mlx"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="yaml file with dataset names",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="current worker in [0, num-downloaders -1]",
    )
    parser.add_argument(
        "--num-downloaders", type=int, default=1, help="number of parallel downloaders"
    )
    parser.add_argument("--no_bandwidth", action="store_true")
    parser.add_argument(
        "--download_tar",
        action="store_true",
        help="whether or not to download tar files also",
    )
    parser.add_argument("--pretrained-text-embeddings", type=str, default=None)
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="",
        help="end point for the s3 bucket — uses environment variable AWS_ENDPOINT_URL otherwise",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="subset to download [train|eval]",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level="INFO",
        format=(
            "[%(asctime)s] {%(pathname)s:%(lineno)d}" "%(levelname)s - %(message)s"
        ),
        datefmt="%H:%M:%S",
    )
    helpers.print_args(args)
    main(args)
