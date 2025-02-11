# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import datetime
import logging
import os

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_singlenode(timeout=0):
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not "MASTER_ADDR" in os.environ or world_size == 1:
        return local_rank, rank, world_size

    if timeout == 0:
        timeout = dist.default_pg_timeout
    else:
        timeout = datetime.timedelta(seconds=timeout)

    logging.info(f"Default timeout: {timeout}")
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        timeout=timeout,
        rank=rank,
    )

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    logging.info(
        f"setting up local_rank {local_rank} global_rank {rank} world size {world_size}"
    )
    setup_for_distributed(rank == 0)
    return local_rank, rank, world_size


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


# ----------------------------------------------------------------------------


def get_world_size():
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
