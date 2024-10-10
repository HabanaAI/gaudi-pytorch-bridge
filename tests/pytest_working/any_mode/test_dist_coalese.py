import argparse
import numbers
import os
from typing import Any, Callable, Dict

import habana_frameworks.torch
import habana_frameworks.torch as ht
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.testing

device = torch.device("hpu")
WORLD_SIZE = habana_frameworks.torch.hpu.device_count()


def get_world_trs():
    return {
        "tag": "",
        "ranks": list(range(WORLD_SIZE)),
        "group_size": WORLD_SIZE,
    }


def setup(rank, world_size=1):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)

    import habana_frameworks.torch.distributed.hccl

    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def coalescing_manager_no_device_init_test(rank, world_size, kwargs):
    try:
        comm_ranks = list(range(world_size))
        pg = dist.new_group(ranks=comm_ranks)
        pg._start_coalescing(torch.device(device))
        tensors = [torch.ones(1, device=device), torch.ones(1, device=device)]
        pg.allreduce(tensors)
        cs = pg._end_coalescing(torch.device(device))
        cs.wait()
        assert 0, "Check HPUinit is done before _start_coalescing"
    except RuntimeError as e:
        pass


def coalescing_manager_no_start_coalese_test(rank, world_size, kwargs):
    try:
        comm_ranks = list(range(world_size))
        pg = dist.new_group(ranks=comm_ranks)
        # pg._start_coalescing(torch.device(device))
        tensors = [torch.ones(1, device=device), torch.ones(1, device=device)]
        pg.allreduce(tensors)
        cs = pg._end_coalescing(torch.device(device))
        cs.wait()
        assert 0, "Check _start_coalescing is done before _end_coalescing"
    except RuntimeError as e:
        pass


def coalescing_manager_test(rank, world_size, kwargs):
    tensors = [torch.ones(1, device=device), torch.ones(1, device=device)]
    comm_ranks = list(range(world_size))
    pg = dist.new_group(ranks=comm_ranks)
    pg._start_coalescing(torch.device(device))
    pg.allreduce(tensors, dist.ReduceOp.SUM)
    cs = pg._end_coalescing(torch.device(device))
    cs.wait()
    for tensor in tensors:
        torch.testing.assert_close(tensor.to("cpu"), torch.ones(1) * world_size)


def allreduce_coalesed_manager_test(rank, world_size, kwargs):
    tensors1 = [torch.ones(1, device=device), torch.ones(1, device=device)]
    tensors2 = [torch.ones(1, device=device), torch.ones(1, device=device)]
    comm_ranks = list(range(world_size))
    pg = dist.new_group(ranks=comm_ranks)
    pg._start_coalescing(torch.device(device))
    pg.allreduce(tensors1)
    pg.allreduce(tensors2)
    cs = pg._end_coalescing(torch.device(device))
    cs.wait()
    for tensor in tensors1:
        torch.testing.assert_close(tensor.to("cpu"), torch.ones(1) * world_size)
    for tensor in tensors2:
        torch.testing.assert_close(tensor.to("cpu"), torch.ones(1) * world_size)


def allreduce_coalesed_test(rank, world_size, kwargs):
    tensors = [torch.ones(10, 10, device=torch.device(device)), torch.ones(10, 10, device=torch.device(device))]
    dist.all_reduce_coalesced(tensors, dist.ReduceOp.SUM)
    for tensor in tensors:
        torch.testing.assert_close(tensor.to("cpu"), torch.ones(10, 10) * world_size)


def reduce_scatter_tensor_coalesced_test(rank, world_size, kwargs):
    comm_ranks = list(range(world_size))
    output_tensors = [torch.empty(10, device=device), torch.empty(10, device=device)]
    input_tensors = [
        torch.ones((tensor.size(0) * world_size, *tensor.size()[1:]), device=tensor.device, dtype=tensor.dtype)
        for tensor in output_tensors
    ]

    pg = dist.new_group(ranks=comm_ranks)
    from torch._C._distributed_c10d import ReduceOp, ReduceScatterOptions

    opts = ReduceScatterOptions()
    opts.reduceOp = ReduceOp.SUM
    pg.reduce_scatter_tensor_coalesced(output_tensors, input_tensors, opts)

    for i, output in enumerate(output_tensors):
        assert output.eq(world_size).all()


def allgather_into_tensor_coalesced_test(rank, world_size, kwargs):
    input_tensors = [torch.ones(1, device=device), torch.ones(1, device=device)]
    output_tensors = [
        torch.empty((tensor.size(0) * world_size, *tensor.size()[1:]), device=tensor.device, dtype=tensor.dtype)
        for tensor in input_tensors
    ]

    comm_ranks = list(range(world_size))
    pg = dist.new_group(ranks=comm_ranks)
    pg._start_coalescing(torch.device(device))
    for output, input in zip(output_tensors, input_tensors):
        torch.distributed.distributed_c10d.all_gather_into_tensor(output, input, group=pg, async_op=kwargs["async_op"])
    cs = pg._end_coalescing(torch.device(device))
    cs.wait()
    k = torch.ones(1 * world_size)
    for tensor in output_tensors:
        torch.testing.assert_close(tensor.to("cpu"), k)


def dynamo_coalescing_manager_test(rank, world_size, kwargs):
    from torch._dynamo.testing import CompileCounter

    inputs = [torch.ones(1, device=device), torch.ones(1, device=device)]
    comm_ranks = list(range(world_size))
    pg = dist.new_group(ranks=comm_ranks)

    def func(t, *, tag, ranks, group_size):
        pg._start_coalescing(torch.device(device))
        pg.allreduce(t, dist.ReduceOp.SUM)
        cs = pg._end_coalescing(torch.device(device))
        cs.wait()
        return t

    counter = CompileCounter()
    compiled = torch.compile(func, backend=counter)
    out = compiled(inputs, **get_world_trs())
    for t in out:
        torch.testing.assert_close(t.to("cpu"), torch.ones(1) * world_size)


def dynamo_trace_allgather_coalesced_test(rank, world_size, kwargs):
    from torch._dynamo.testing import CompileCounter

    device = torch.device("hpu")

    def func(inp, *, tag, ranks, group_size):
        ar = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(inp, tag, ranks, group_size)
        return ar

    inputs = [torch.ones(4, 4, device="hpu"), torch.ones(6, 6, device="hpu")]
    counter = CompileCounter()
    compiled = torch.compile(func, backend=counter)
    out = compiled(inputs, **get_world_trs())
    assert counter.frame_count == 1
    assert counter.op_count == 3  # It generates 2 getattr to unpack the array
    expected_outputs = [torch.ones(4 * world_size, 4), torch.ones(6 * world_size, 6)]
    for i, tensor in enumerate(out):
        torch.testing.assert_close(tensor.to("cpu"), expected_outputs[i])


def batch_isend_irecv_hccl_test(rank, world_size, kwargs):
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


def run_test(rank: int, world_size: int, test_func: Callable, kwargs):
    setup(rank, world_size)
    if rank == 0:
        print("Running test :", test_func, kwargs)
    test_func(rank, world_size, kwargs)
    cleanup()


def run_tests():
    test_configs = [
        {"func": coalescing_manager_no_device_init_test, "kwargs": {}},
        {"func": coalescing_manager_no_start_coalese_test, "kwargs": {}},
        {"func": coalescing_manager_test, "kwargs": {}},
        {"func": allreduce_coalesed_manager_test, "kwargs": {}},
        {"func": allreduce_coalesed_test, "kwargs": {}},
        {"func": allgather_into_tensor_coalesced_test, "kwargs": {"async_op": True}},
        {"func": allgather_into_tensor_coalesced_test, "kwargs": {"async_op": False}},
        {"func": reduce_scatter_tensor_coalesced_test, "kwargs": {}},
        {"func": dynamo_coalescing_manager_test, "kwargs": {}},
        {"func": dynamo_trace_allgather_coalesced_test, "kwargs": {}},
        {"func": batch_isend_irecv_hccl_test, "kwargs": {}},
    ]
    for config in test_configs:
        mp.spawn(run_test, args=(WORLD_SIZE, config["func"], config["kwargs"]), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    run_tests()
