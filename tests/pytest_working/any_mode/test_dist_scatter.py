import argparse
import os
from typing import List

import habana_frameworks.torch
import habana_frameworks.torch as ht
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device_hpu = torch.device("hpu")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    import habana_frameworks.torch.distributed.hccl

    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def simple(rank, world_size, args):
    print("rank :: ", rank)
    print("world_size :: ", world_size)
    device = f"{device_hpu}:{rank}"
    setup(rank, world_size)
    if rank == 0:
        scatter_list = [torch.ones(3, 3, device="hpu") * rank for rank in range(world_size)]
    else:
        scatter_list = None
    output_tensor = torch.empty(3, 3, device="hpu")
    dist.scatter(output_tensor, scatter_list)
    result_cmp = torch.ones(3, 3, device="hpu") * rank
    result = torch.all(output_tensor.eq(result_cmp))
    assert result.item() == True
    print("DONE for rank :: ", rank)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_reduce_scatter")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    args = parser.parse_args()
    if args.verbose:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    WORLD_SIZE = habana_frameworks.torch.hpu.device_count()
    if WORLD_SIZE > 1:
        mp.spawn(simple, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
