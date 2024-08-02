# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3.session
from boto3.s3.transfer import TransferConfig

ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", None)


def download_object(
    bucket_name,
    file_name,
    download_path=None,
    endpoint_url=ENDPOINT_URL,
    max_bandwidth=None,
):
    """Downloads an object from S3 to local."""
    session = boto3.session.Session()
    s3_client = session.client(service_name="s3", endpoint_url=endpoint_url)
    if download_path is None:
        download_path = os.path.basename(file_name)
    try:
        s3_client.download_file(
            bucket_name,
            file_name,
            download_path,
            Config=TransferConfig(
                num_download_attempts=10, max_bandwidth=max_bandwidth
            ),
        )
    except Exception as e:
        logging.error(f"Error thrown while downloading file: {file_name}. {e}")
    return download_path


def download_object_from_full_path(path, download_path=None, endpoint_url=ENDPOINT_URL):
    bucket_name, parent_path, basename = _parse_path(path)
    file_name = os.path.join(parent_path, basename)
    return download_object(
        bucket_name, file_name, download_path=download_path, endpoint_url=endpoint_url
    )


def upload_object(
    bucket_name,
    file_name,
    upload_path,
    endpoint_url=ENDPOINT_URL,
):
    """Uload an object from S3 to local."""

    session = boto3.session.Session()
    s3_client = session.client(service_name="s3", endpoint_url=endpoint_url)
    s3_client.upload_file(file_name, bucket_name, upload_path)
    return "Success"


def _parse_path(tsv_pattern):
    # for now, expected path is s3://mlx/datasets/{datset_name}/rest
    parts = tsv_pattern.split("/")
    assert parts[0] == "s3:"
    assert parts[1] == ""
    assert parts[3] == "datasets"  # for now, since everything is in mlx/datasets
    bucket = parts[2]
    pattern = parts[-1]
    return bucket, "/".join(parts[3:-1]), pattern


def get_file_list(tsv_pattern, endpoint_url=ENDPOINT_URL):
    bucket_name, parent_path, pattern = _parse_path(tsv_pattern)
    resource = boto3.resource("s3", endpoint_url=endpoint_url)
    bucket = resource.Bucket(bucket_name)
    fnames = []
    pattern = re.compile(pattern)
    for obj in bucket.objects.filter(Prefix=parent_path + "/"):
        fname = obj.key
        if pattern.search(fname):
            fnames.append(f"s3://{bucket_name}/{fname}")

    return fnames


def download_parallel(files, endpoint_url=ENDPOINT_URL):
    logging.info("Doing parallel download")
    with ProcessPoolExecutor() as executor:
        logging.info(f"Submitting {files}")
        future_to_key = {
            executor.submit(
                download_object_from_full_path,
                key,
                f"{index}.tsv",
                endpoint_url=endpoint_url,
            ): key
            for index, key in enumerate(files)
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()
            if not exception:
                yield key, future.result()
            else:
                yield key, exception
