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
from copy import deepcopy
from typing import List

import habana_frameworks.torch.hpu

# import habana_frameworks.torch.low_overhead_profiler.profiler as lop
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

"""
schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

profiler = torch.profiler.profile(
    schedule=schedule,
    activities=activities,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./"),
    record_shapes=False,
    with_stack=True,
)
"""


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    import habana_frameworks.torch.distributed.hccl

    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    # profiler.start()
    # lop.start()


def cleanup():
    dist.destroy_process_group()
    # profiler.stop()
    # lop.stop()
    # lop.flush()


device_hpu = torch.device("hpu")


def all_gather_with_odd_size(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    # test all_gather
    input_tensor = torch.ones(63, device=device_hpu, dtype=torch.uint8) * world_size
    output_tensor_list = [torch.zeros(63, device=device_hpu, dtype=torch.uint8) for _ in range(world_size)]
    dist.all_gather(output_tensor_list, input_tensor, async_op=True).wait()

    for tensor in output_tensor_list:
        torch.testing.assert_close(tensor, input_tensor)

    dist.barrier()
    cleanup()


def all_gather_into_tensor_with_odd_size(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    # test all_gather_into_tensor
    input_tensor = torch.arange(63, device=device_hpu, dtype=torch.uint8) + (rank * 63)
    output_tensor = torch.zeros(63 * world_size, device=device_hpu, dtype=torch.uint8)
    dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True).wait()

    torch.testing.assert_close(torch.arange(63 * world_size, device=device_hpu, dtype=torch.uint8), output_tensor)

    dist.barrier()
    cleanup()


def broadcast_with_odd_size(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    # test broadcast
    if rank == 0:
        input_tensor = torch.ones(99, 99, device=device_hpu, dtype=torch.uint8)
    else:
        input_tensor = torch.zeros(99, 99, device=device_hpu, dtype=torch.uint8)
    dist.broadcast(input_tensor, 0, async_op=True).wait()
    torch.testing.assert_close(torch.ones(99, 99, device=device_hpu, dtype=torch.uint8), input_tensor)

    dist.barrier()
    cleanup()


def all_gather_with_odd_size_and_view(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    # test all_gather
    input_tensor = torch.arange(81, dtype=torch.float).as_strided((9, 9), (1, 9)).to(device)
    output_tensor_list = [
        torch.zeros(81, dtype=torch.float).as_strided((9, 9), (1, 9)).to(device) for _ in range(world_size)
    ]
    dist.all_gather(output_tensor_list, input_tensor, async_op=True).wait()

    for tensor in output_tensor_list:
        torch.testing.assert_close(tensor, input_tensor)

    dist.barrier()
    cleanup()


def simple(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    # test all_gather
    input_tensor = torch.ones(100, 100, device=device_hpu) * 7
    output_tensor_list = [torch.zeros(100, 100, device=device_hpu) for _ in range(world_size)]
    dist.all_gather(output_tensor_list, input_tensor, async_op=True).wait()

    for tensor in output_tensor_list:
        torch.testing.assert_close(tensor, input_tensor)

    # test all_gather_into_tensor
    input_tensor = torch.arange(100, device=device_hpu, dtype=torch.float) + (rank * 100)
    output_tensor = torch.zeros(100 * world_size, device=device_hpu, dtype=torch.float)
    dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True).wait()

    torch.testing.assert_close(torch.arange(100 * world_size, device=device_hpu, dtype=torch.float), output_tensor)

    # test all_reduce
    input_tensor = torch.ones(100, 100, device=device_hpu) * 7
    dist.all_reduce(input_tensor, async_op=True).wait()
    torch.testing.assert_close(input_tensor, torch.ones(100, 100, device=device_hpu) * (7 * world_size))

    # test broadcast
    if rank == 0:
        input_tensor = torch.ones(100, 100, device=device_hpu)
    else:
        input_tensor = torch.zeros(100, 100, device=device_hpu)
    dist.broadcast(input_tensor, 0, async_op=True).wait()
    torch.testing.assert_close(torch.ones(100, 100, device=device_hpu), input_tensor)

    # test reduce_scatter
    output_tensor = torch.zeros(100, 100, device=device_hpu)
    input_tensor_list = [torch.ones(100, 100, device=device_hpu) for _ in range(world_size)]
    dist.reduce_scatter(output_tensor, input_tensor_list, async_op=True).wait()
    torch.testing.assert_close(output_tensor, torch.zeros(100, 100, device=device_hpu) + world_size)

    dist.barrier()
    cleanup()


def send_recieve_with_odd_size(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    def local_send(nodes):
        _tensor = torch.ones(9, 9, device=device_hpu).to(torch.int8)
        for node in nodes:
            torch.distributed.send(_tensor, node)

    def local_recv():
        _tensor = torch.ones(9, 9, device=device_hpu).to(torch.int8)
        torch.distributed.recv(_tensor, 0)
        return _tensor

    _tensor_ref = torch.ones(9, 9, device=device_hpu).to(torch.int8)
    if rank == 0:
        local_send(range(1, world_size))
    else:
        _tensor = local_recv()
        assert torch.equal(_tensor, _tensor_ref)


def send_recieve(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    def local_send(nodes):
        _tensor = torch.ones(100, 100, device=device_hpu)
        for node in nodes:
            torch.distributed.send(_tensor, node)

    def local_recv():
        _tensor = torch.zeros(100, 100, device=device_hpu)
        torch.distributed.recv(_tensor, 0)
        return _tensor

    _tensor_ref = torch.ones(100, 100, device=device_hpu)
    if rank == 0:
        local_send(range(1, world_size))
    else:
        _tensor = local_recv()
        assert torch.equal(_tensor, _tensor_ref)


def send_recieve_permuted(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    def local_send(tensor, nodes):
        # _tensor = torch.ones(100, 100, device=device_hpu)
        for node in nodes:
            torch.distributed.send(tensor, node)

    def local_recv():
        _tensor = torch.zeros(1, 3, 3, 3, dtype=torch.float32, device=device_hpu)
        torch.distributed.recv(_tensor, 0)
        return _tensor

    input_a = torch.arange(18, dtype=torch.float32, requires_grad=False).reshape(1, 2, 3, 3).to("hpu")
    weight_a = torch.arange(6, dtype=torch.float32, requires_grad=False).reshape(3, 2, 1, 1).to("hpu")
    conv = torch.nn.functional.conv2d(input_a, weight_a, bias=None, stride=1, padding=0, dilation=1, groups=1)
    conv_cpu = conv.to("cpu")

    if rank == 0:
        # cache miss for copy d2d for sendPermuteToDense
        local_send(conv, range(1, world_size))
        # print("send1 tensor_cpu", conv_cpu)

        # cache hit for copy d2d for sendPermuteToDense
        local_send(conv, range(1, world_size))
        # print("send2 conv_cpu", conv_cpu)
    else:
        _tensor = local_recv()
        # print("recv1 tensor_cpu", _tensor.to("cpu"))
        assert torch.equal(_tensor.to("cpu"), conv_cpu)

        _tensor = local_recv()
        # print("recv2 conv_cpu", _tensor.to("cpu"))
        assert torch.equal(_tensor.to("cpu"), conv_cpu)


# gather collective uses send/receive
def gather_with_odd_size(rank, world_size, args):
    device = f"{device_hpu}"
    setup(rank, world_size)

    input = rank * torch.ones((15), dtype=torch.uint8, device=device)
    output_list = [torch.empty_like(input, dtype=torch.uint8) for _ in range(world_size)]
    expected_output_list = [i * torch.ones_like(input, dtype=torch.uint8, device=device) for i in range(world_size)]

    for r in range(world_size):
        torch.distributed.gather(input, output_list if rank == r else None, dst=r)
        if rank == r:
            for t1, t2 in zip(expected_output_list, output_list):
                assert torch.equal(t1.cpu(), t2.cpu()), (
                    f"Gathered tensor is not equal to expected one. " f"Got: {t2}, expected: {t1}."
                )

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_eager_collective_asycn test for veriying async op")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    args = parser.parse_args()
    if args.verbose:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    WORLD_SIZE = habana_frameworks.torch.hpu.device_count()
    if WORLD_SIZE > 1:
        mp.spawn(simple, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)

    if WORLD_SIZE > 2:
        mp.spawn(all_gather_into_tensor_with_odd_size, args=(2, args), nprocs=2, join=True)
        mp.spawn(all_gather_into_tensor_with_odd_size, args=(3, args), nprocs=3, join=True)
        mp.spawn(broadcast_with_odd_size, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
        mp.spawn(all_gather_with_odd_size, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
        mp.spawn(all_gather_with_odd_size_and_view, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
        mp.spawn(send_recieve, args=(2, args), nprocs=2, join=True)
        mp.spawn(send_recieve_with_odd_size, args=(2, args), nprocs=2, join=True)
        mp.spawn(send_recieve_permuted, args=(2, args), nprocs=2, join=True)
        mp.spawn(gather_with_odd_size, args=(2, args), nprocs=2, join=True)
