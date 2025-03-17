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


def simple(rank, world_size, args):
    print("rank :: ", rank)
    print("world_size :: ", world_size)
    device = f"{device_hpu}:{rank}"
    setup(rank, world_size)
    ######################reduce_scatter test
    to_reduce_scatter = [torch.ones(3, 3, device="hpu") * rank for rank in range(world_size)]
    output_tensor = torch.empty(3, 3, device="hpu")

    dist.reduce_scatter(output_tensor, to_reduce_scatter)
    expected_tensor = torch.ones(3, 3) * dist.get_rank() * world_size

    output_tensor = torch.empty(3, 3, device="hpu")
    dist.reduce_scatter(output_tensor, to_reduce_scatter, op=dist.ReduceOp.AVG)
    expected_tensor = torch.ones(3, 3, device="hpu") * dist.get_rank()
    if torch.allclose(output_tensor, expected_tensor, rtol=1e-5, atol=1e-5):
        print(f"Process {rank}: reduce_scatter Test passed!")
    else:
        assert f"Process {rank}: reduce_scatter Test failed! Expected {expected_tensor}, got {output_tensor}"
    #######################reduce test
    # Each process has a tensor
    data = torch.tensor([rank + 1.0], dtype=torch.float32, device="hpu")

    # Perform reduce with AVG
    dist.reduce(data, dst=0, op=dist.ReduceOp.AVG)

    # Verify the result on the root process (rank 0)
    if rank == 0:
        expected_result = torch.tensor(
            [(sum(range(1, world_size + 1))) / world_size], dtype=torch.float32, device="hpu"
        )
        if torch.allclose(data, expected_result, rtol=1e-5, atol=1e-5):
            print(f"Process {rank}: reduce Test passed! Result: {data}")
        else:
            print(f"Process {rank}: reduce Test failed! Expected {expected_result}, got {data}")
    #######################all_reduce test
    # Each process has a tensor
    data = torch.tensor([rank + 1.0], dtype=torch.float32, device="hpu")

    # Perform allreduce with AVG
    dist.all_reduce(data, op=dist.ReduceOp.AVG)

    # Expected result for all processes
    expected_result = torch.tensor([(sum(range(1, world_size + 1))) / world_size], dtype=torch.float32, device="hpu")

    # Verify the result
    if torch.allclose(data, expected_result, rtol=1e-5, atol=1e-5):
        print(f"Process {rank}: all_reduce Test passed! Result: {data}")
    else:
        print(f"Process {rank}: all_reduce Test failed! Expected {expected_result}, got {data}")

    #######################reduce_scatter_tensor test
    # Each process has a tensor of 4 elements
    data = torch.tensor([rank + i + 1 for i in range(world_size)], dtype=torch.float32, device="hpu")
    recv_data = torch.zeros(1, dtype=torch.float32, device="hpu")  # Each process will receive 1 element after reduction

    # Perform reduce_scatter with AVG
    dist.reduce_scatter_tensor(recv_data, data, op=dist.ReduceOp.AVG)
    expected_result = torch.tensor(
        [(sum(range(1, world_size + 1)) + world_size * rank) / world_size], dtype=torch.float32, device="hpu"
    )
    if torch.allclose(recv_data, expected_result, rtol=1e-5, atol=1e-5):
        print(f"Process {rank}: reduce_scatter_tensor Test passed!")
    else:
        assert f"Process {rank}:reduce_scatter_tensor Test failed! Expected {expected_result}, got {recv_data}"

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
