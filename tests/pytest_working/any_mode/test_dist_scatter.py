###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import argparse
import os

import habana_frameworks.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

device_hpu = torch.device("hpu")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def simple(rank, world_size, dtype):
    print("rank :: ", rank)
    print("world_size :: ", world_size)
    device = f"{device_hpu}:{rank}"
    setup(rank, world_size)
    if rank == 0:
        scatter_list = [torch.ones(3, 3, device="hpu", dtype=dtype) * rank for rank in range(world_size)]
    else:
        scatter_list = None
    output_tensor = torch.empty(3, 3, device="hpu", dtype=dtype)
    dist.scatter(output_tensor, scatter_list)
    result_cmp = torch.ones(3, 3, device="hpu", dtype=dtype) * rank
    result = torch.all(output_tensor.eq(result_cmp))
    assert result.item() is True
    print("DONE for rank :: ", rank)
    cleanup()


def empty_tensor(rank, world_size, dtype):
    device = f"{device_hpu}:{rank}"
    setup(rank, world_size)
    if rank == 0:
        scatter_list = [torch.ones(0, device="hpu", dtype=dtype) * rank for rank in range(world_size)]
    else:
        scatter_list = None
    output_tensor = torch.empty(0, device="hpu", dtype=dtype)
    dist.scatter(output_tensor, scatter_list)
    result_cmp = torch.ones(0, device="hpu", dtype=dtype) * rank
    result = torch.all(output_tensor.eq(result_cmp))
    assert result.item() is True
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
        mp.spawn(simple, args=(WORLD_SIZE, torch.float32), nprocs=WORLD_SIZE, join=True)
        mp.spawn(simple, args=(WORLD_SIZE, torch.int8), nprocs=WORLD_SIZE, join=True)
        os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"
        mp.spawn(simple, args=(WORLD_SIZE, torch.float32), nprocs=WORLD_SIZE, join=True)
        mp.spawn(simple, args=(WORLD_SIZE, torch.int8), nprocs=WORLD_SIZE, join=True)
        mp.spawn(empty_tensor, args=(WORLD_SIZE, torch.float32), nprocs=WORLD_SIZE, join=True)
