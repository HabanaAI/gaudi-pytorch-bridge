###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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
    output_tensor = torch.zeros(2, dtype=torch.float16).to(device)
    input_tensor = torch.arange(world_size * 2, dtype=torch.float16).to(device)
    input_tensor = torch.reshape(input_tensor, (world_size, 2))
    dist.reduce_scatter_tensor(output_tensor, input_tensor)
    result_cmp = input_tensor * world_size
    for i, j in zip(result_cmp[rank], output_tensor):
        if i != j:
            assert "output value is incorrect"
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
