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


import os
from typing import Any, Callable, Dict

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl
import habana_frameworks.torch.hpu
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

device_hpu = torch.device("hpu")
WORLD_SIZE = habana_frameworks.torch.hpu.device_count()

skip_test = True


# Function to set up the distributed process group
def init_hccl(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"
    # Initialize the process group
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def no_device_init_test(rank, world_size, coalescing):
    init_hccl(rank, world_size)
    try:
        comm_ranks = list(range(world_size))
        pg = dist.new_group(ranks=comm_ranks)
        pg._start_coalescing(torch.device(device_hpu))
        tensors = [torch.ones(1, device=device_hpu), torch.ones(1, device=device_hpu)]
        pg.allreduce(tensors)
        cs = pg._end_coalescing(torch.device(device_hpu))
        cs.wait()
        assert 0, "Check HPUinit is done before _start_coalescing"
    except Exception as e:
        pass

    dist.barrier()
    cleanup()


def no_start_coalese_test(rank, world_size, coalescing):
    init_hccl(rank, world_size)
    try:
        comm_ranks = list(range(world_size))
        pg = dist.new_group(ranks=comm_ranks)

        tensors = [torch.ones(1, device=device_hpu), torch.ones(1, device=device_hpu)]
        pg.allreduce(tensors)
        cs = pg._end_coalescing(torch.device(device_hpu))
        cs.wait()
        assert 0, "Check _start_coalescing is done before _end_coalescing"
    except Exception as e:
        pass
        dist.barrier()
        cleanup()


def reduce_scatter_tensor_coalesced_test(rank, world_size, coalescing):
    init_hccl(rank, world_size)
    comm_ranks = list(range(world_size))
    output_tensors = [torch.empty(10, device=device_hpu), torch.empty(10, device=device_hpu)]
    input_tensors = [
        torch.ones((tensor.size(0) * world_size, *tensor.size()[1:]), device=tensor.device, dtype=tensor.dtype)
        for tensor in output_tensors
    ]

    pg = dist.new_group(ranks=comm_ranks)
    if coalescing:
        pg._start_coalescing(torch.device(device_hpu))

    from torch._C._distributed_c10d import ReduceOp, ReduceScatterOptions

    opts = ReduceScatterOptions()
    opts.reduceOp = ReduceOp.SUM
    pg.reduce_scatter_tensor_coalesced(output_tensors, input_tensors, opts)

    if coalescing:
        cs = pg._end_coalescing(torch.device(device_hpu))
        cs.wait()

    for i, output in enumerate(output_tensors):
        assert output.eq(world_size).all()

    dist.barrier()
    cleanup()


def allgather_into_tensor_coalesced_test(rank, world_size, coalescing):
    init_hccl(rank, world_size)
    input_tensors = [torch.ones(1, device=device_hpu), torch.ones(1, device=device_hpu)]
    output_tensors = [
        torch.empty((tensor.size(0) * world_size, *tensor.size()[1:]), device=tensor.device, dtype=tensor.dtype)
        for tensor in input_tensors
    ]

    comm_ranks = list(range(world_size))
    pg = dist.new_group(ranks=comm_ranks)

    if coalescing:
        pg._start_coalescing(torch.device(device_hpu))

    for output, input in zip(output_tensors, input_tensors):
        torch.distributed.distributed_c10d.all_gather_into_tensor(output, input, group=pg)

    if coalescing:
        cs = pg._end_coalescing(torch.device(device_hpu))
        cs.wait()

    k = torch.ones(1 * world_size)
    for tensor in output_tensors:
        torch.testing.assert_close(tensor.to("cpu"), k)

    dist.barrier()
    cleanup()


def all_reduce(rank, world_size, coalescing):
    init_hccl(rank, world_size)
    tensors = [torch.ones(1, device=device_hpu), torch.ones(1, device=device_hpu)]
    comm_ranks = list(range(world_size))
    pg = dist.new_group(ranks=comm_ranks)

    if coalescing:
        pg._start_coalescing(device_hpu)

    pg.allreduce(tensors, dist.ReduceOp.SUM)

    if coalescing:
        cs = pg._end_coalescing(device_hpu)
        cs.wait()

    for tensor in tensors:
        torch.testing.assert_close(tensor.to("cpu"), torch.ones(1) * world_size)

    cleanup()


def batch_isend_irecv_hccl_test(rank, world_size, coalescing):
    init_hccl(rank, world_size)

    def _build_tensor(size, value=None, dtype=torch.bfloat16):
        if value is None:
            value = size
        return torch.empty(size, 1, 4096, dtype=dtype).fill_(value).to("hpu")

    p2p_op_list = []
    for src in range(0, world_size):
        send_tensor = _build_tensor(rank + 1)
        recv_tensor = _build_tensor(src + 1)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, src)
        p2p_op_list.append(recv_op)
        send_op = dist.P2POp(dist.isend, send_tensor, src)
        p2p_op_list.append(send_op)

    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()

    dist.barrier()
    cleanup()


def simple_all_gather(rank, world_size):
    init_hccl(rank, world_size)

    # test all_gather
    input_tensor = torch.ones(100, 100, device=device_hpu) * 7
    output_tensor_list = [torch.zeros(100, 100, device=device_hpu) for _ in range(world_size)]
    dist.all_gather(output_tensor_list, input_tensor, async_op=True).wait()

    for tensor in output_tensor_list:
        torch.testing.assert_close(tensor, input_tensor)

    dist.barrier()
    cleanup()


def simple_all_gather_into_tensor(rank, world_size):
    init_hccl(rank, world_size)
    # test all_gather_into_tensor
    input_tensor = torch.arange(100, device=device_hpu, dtype=torch.float) + (rank * 100)
    output_tensor = torch.zeros(100 * world_size, device=device_hpu, dtype=torch.float)
    dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True).wait()

    torch.testing.assert_close(torch.arange(100 * world_size, device=device_hpu, dtype=torch.float), output_tensor)

    dist.barrier()
    cleanup()


def simple_all_reduce(rank, world_size):
    init_hccl(rank, world_size)
    # test all_reduce
    input_tensor = torch.ones(100, 100, device=device_hpu) * 7
    dist.all_reduce(input_tensor, async_op=True).wait()
    torch.testing.assert_close(input_tensor, torch.ones(100, 100, device=device_hpu) * (7 * world_size))

    dist.barrier()
    cleanup()


def simple_broadcast(rank, world_size):
    init_hccl(rank, world_size)
    # test broadcast
    if rank == 0:
        input_tensor = torch.ones(100, 100, device=device_hpu)
    else:
        input_tensor = torch.zeros(100, 100, device=device_hpu)
    dist.broadcast(input_tensor, 0, async_op=True).wait()
    torch.testing.assert_close(torch.ones(100, 100, device=device_hpu), input_tensor)

    dist.barrier()
    cleanup()


def simple_reduce_scatter(rank, world_size):
    init_hccl(rank, world_size)
    # test reduce_scatter
    output_tensor = torch.zeros(100, 100, device=device_hpu)
    input_tensor_list = [torch.ones(100, 100, device=device_hpu) for _ in range(world_size)]
    dist.reduce_scatter(output_tensor, input_tensor_list, async_op=True).wait()
    torch.testing.assert_close(output_tensor, torch.zeros(100, 100, device=device_hpu) + world_size)

    dist.barrier()
    cleanup()


@pytest.mark.parametrize(
    "method",
    [simple_all_gather, simple_all_gather_into_tensor, simple_all_reduce, simple_broadcast, simple_reduce_scatter],
)
@pytest.mark.parametrize("world_size", [WORLD_SIZE])
@pytest.mark.skipif(skip_test, reason="SW-205798")
def test_(method, world_size):
    mp.spawn(method, args=(world_size,), nprocs=world_size)


@pytest.mark.parametrize(
    "method",
    [all_reduce, no_device_init_test, no_start_coalese_test, batch_isend_irecv_hccl_test],
)
@pytest.mark.parametrize("world_size", [WORLD_SIZE])
@pytest.mark.parametrize("coalescing", [True, False])
@pytest.mark.skipif(skip_test, reason="SW-205798")
def test_coalescing_manager_(method, world_size, coalescing):
    mp.spawn(method, args=(world_size, coalescing), nprocs=world_size)


def run_tests():
    test_configs = [
        {"func": simple_all_gather},
        {"func": simple_all_gather_into_tensor},
        {"func": simple_all_reduce},
        {"func": simple_broadcast},
        {"func": simple_reduce_scatter},
    ]

    test_coalescing_configs = [
        {"func": all_reduce, "coalescing": True},
        {"func": all_reduce, "coalescing": False},
        {"func": no_device_init_test, "coalescing": True},
        {"func": no_start_coalese_test, "coalescing": True},
        {"func": reduce_scatter_tensor_coalesced_test, "coalescing": True},
        {"func": allgather_into_tensor_coalesced_test, "coalescing": True},
        {"func": batch_isend_irecv_hccl_test, "coalescing": True},
    ]
    for config in test_configs:
        mp.spawn(config["func"], args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    for config in test_coalescing_configs:
        mp.spawn(config["func"], args=(WORLD_SIZE, config["coalescing"]), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    run_tests()
