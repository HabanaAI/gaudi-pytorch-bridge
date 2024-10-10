# pytest testCollectives.py -sv

import cProfile
import os
import pstats
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch.manual_seed(0)
device = torch.device("hpu")

ITER = 10
TENSOR_LEN = 1024 * 1024 * 1


class DistSetup:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12340"
        import habana_frameworks.torch.distributed.hccl

        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        # Following Code is to ensure that HCL_Init is done
        _tensor = torch.ones(1).to(device)
        torch.distributed.all_reduce(_tensor)

    def __del__(self):
        dist.destroy_process_group()


# Test: Functional test of synchronous Reduce
def sync_reduce_func(hpu, world_size, dtype):
    _tensor_ref = world_size * torch.ones(TENSOR_LEN).to(dtype).to(device)

    for i in range(ITER):
        src_hpu = i % world_size
        _tensor = torch.ones(TENSOR_LEN).to(dtype).to(device)
        # Only Rank-0 will receive the result
        torch.distributed.reduce(_tensor, src_hpu)
        if hpu == src_hpu:
            assert torch.equal(_tensor, _tensor_ref) and _tensor.dtype == dtype


# Test: Functional test of synchronous AllGather
def sync_allGather_func(hpu, world_size, dtype):
    _tensor_ref = [i * torch.ones(TENSOR_LEN).to(dtype).to(device) for i in range(world_size)]

    for i in range(ITER):
        _tensor_list = [torch.zeros(TENSOR_LEN).to(dtype).to(device) for _ in range(world_size)]

        _tensor = hpu * torch.ones(TENSOR_LEN).to(dtype).to(device)
        torch.distributed.all_gather(_tensor_list, _tensor)
        for t1, t2 in zip(_tensor_ref, _tensor_list):
            assert torch.equal(t1.cpu(), t2.cpu()) and t1.dtype == dtype and t1.dtype == t2.dtype


def sync_allGather_func_uneven_size(hpu, world_size, dtype):
    output_split_sizes = []
    group = list(range(0, world_size))
    for dst in group:
        output_split_sizes.append(dst + 1)
    sum_len = sum(output_split_sizes)
    _tensor_ref = torch.ones(sum_len, sum_len).to(dtype).to(device)
    for i in range(ITER):
        tensor = torch.ones(output_split_sizes[hpu], sum_len).to(dtype).to(device)
        out_tensor = torch.zeros(sum_len, sum_len).to(dtype).to(device)
        torch.distributed.all_gather(list(torch.split(out_tensor, output_split_sizes)), tensor)
        for t1, t2 in zip(_tensor_ref, out_tensor):
            assert torch.equal(t1.cpu(), t2.cpu()) and t1.dtype == dtype and t1.dtype == t2.dtype


def async_allGather_func_uneven_size(hpu, world_size, dtype):
    output_split_sizes = []
    group = list(range(0, world_size))
    for dst in group:
        output_split_sizes.append(dst + 1)
    sum_len = sum(output_split_sizes)
    _tensor_ref = torch.ones(sum_len, sum_len).to(dtype).to(device)
    for i in range(ITER):
        tensor = torch.ones(output_split_sizes[hpu], sum_len).to(dtype).to(device)
        out_tensor = torch.zeros(sum_len, sum_len).to(dtype).to(device)
        handle = torch.distributed.all_gather(list(torch.split(out_tensor, output_split_sizes)), tensor, async_op=True)
        handle.wait()
        for t1, t2 in zip(_tensor_ref, out_tensor):
            assert torch.equal(t1.cpu(), t2.cpu()) and t1.dtype == dtype and t1.dtype == t2.dtype


# Test: Functional test of synchronous Reduce
def sync_reduceScatter_func(hpu, world_size, dtype):
    for i in range(ITER):
        _tensor_inp = list(torch.ones(TENSOR_LEN * world_size).to(dtype).to(device).chunk(world_size))
        _tensor_out = torch.empty_like(_tensor_inp[hpu]).to(dtype).to(device)
        # Only Rank-0 will receive the result
        torch.distributed.reduce_scatter(_tensor_out, _tensor_inp)


# Test: Functional test of synchronous AllReduce
def sync_allReduce_func(hpu, world_size, dtype):
    _tensor_ref = world_size * torch.ones(TENSOR_LEN).to(dtype).to(device)

    for i in range(ITER):
        _tensor = torch.ones(TENSOR_LEN).to(dtype).to(device)
        torch.distributed.all_reduce(_tensor)
        assert torch.equal(_tensor, _tensor_ref) and _tensor.dtype == dtype


# Test: Testing tensor lifetime of synchronous AllReduce
def sync_allReduce_tensorLife(hpu, world_size, dtype):
    def local_func():
        _tensor = torch.ones(TENSOR_LEN).to(dtype).to(device)
        torch.distributed.all_reduce(_tensor)

    for i in range(ITER):
        local_func()


# Test: Testing tensor lifetime of asynchronous AllReduce
def async_allReduce_tensorLife(hpu, world_size, dtype):
    def local_func():
        _tensor = torch.ones(TENSOR_LEN).to(dtype).to(device)
        handle = torch.distributed.all_reduce(_tensor, async_op=True)
        handle.wait()

    for i in range(ITER):
        local_func()


# Test: Functional test of synchronous Broadcast
def sync_broadcast_func(hpu, world_size, dtype):
    _tensor_ref = torch.ones(TENSOR_LEN).to(dtype).to(device)

    for i in range(ITER):
        _tensor = (hpu + 1) * torch.ones(TENSOR_LEN).to(dtype).to(device)
        torch.distributed.broadcast(_tensor, 0)
        assert torch.equal(_tensor, _tensor_ref) and _tensor.dtype == dtype


# Test: Testing tensor lifetime of synchronous Broadcast
def sync_broadcast_tensorLife(hpu, world_size, dtype):
    def local_func():
        _tensor = torch.ones(TENSOR_LEN).to(dtype).to(device)
        torch.distributed.broadcast(_tensor, 0)

    for i in range(ITER):
        local_func()


# Test: Testing tensor lifetime of asynchronous Broadcast
def async_broadcast_tensorLife(hpu, world_size, dtype):
    def local_func():
        _tensor = torch.ones(TENSOR_LEN).to(dtype).to(device)
        handle = torch.distributed.broadcast(_tensor, 0, async_op=True)
        handle.wait()

    for i in range(ITER):
        local_func()


# Test: Basic functionality of Barrier
def barrier_func(hpu, world_size, dtype):
    for i in range(ITER):
        torch.distributed.barrier()
        # TODO: Need to figure out a way to validate this


# Test: Basic functionality of Send and Receive
def sendReceive_func(hpu, world_size, dtype):
    def local_send(nodes):
        _tensor = torch.ones(TENSOR_LEN).to(dtype).to(device)
        for node in nodes:
            torch.distributed.send(_tensor, node)

    def local_recv():
        _tensor = torch.zeros(TENSOR_LEN).to(dtype).to(device)
        torch.distributed.recv(_tensor, 0)
        return _tensor

    _tensor_ref = torch.ones(TENSOR_LEN).to(dtype).to(device)
    for i in range(ITER):
        if hpu == 0:
            local_send(range(1, world_size))
        else:
            _tensor = local_recv()
            assert torch.equal(_tensor, _tensor_ref) and _tensor.dtype == dtype


# Test: Compute-AllReduce-Compute Perf
# 10 -> TENSOR_LEN -> allReduce -> 10
def computeAllreduceCompute_perf(hpu, world_size, dtype):
    class Model(torch.nn.Module):
        def __init__(self, inp_size, out_size):
            super(Model, self).__init__()
            self.Linear1 = torch.nn.Linear(inp_size, out_size)

        def forward(self, inp):
            return self.Linear1(inp)

    model1 = Model(10, TENSOR_LEN).to(device)
    model2 = Model(TENSOR_LEN, 10).to(device)

    input = torch.ones(10).to(dtype).to(device)
    for i in range(ITER):
        output = model1(input)
        torch.distributed.all_reduce(output)
        output = model2(output)
        _ = output.cpu()


# Test: Host barrier is must have to prevent collectives timeout
def hostsync_broadcast(hpu, world_size, dtype):
    for i in range(5):
        if hpu == 0:
            time.sleep(i + 40)

        # Barrier is must have or else we get timeout
        # torch.distributed.barrier()
        _tensor = torch.ones(TENSOR_LEN).to(device)
        torch.distributed.broadcast(_tensor, 0)


# Test: Functional test of synchronous Gather
def sync_gather_func(hpu, world_size, dtype):
    _tensor_ref = [i * torch.ones(TENSOR_LEN).to(dtype).to(device) for i in range(world_size)]

    _tensor_list = [torch.ones(TENSOR_LEN).to(dtype).to(device) for _ in range(world_size)]

    _tensor = hpu * torch.ones(TENSOR_LEN).to(dtype).to(device)

    if torch.distributed.get_rank() == 0:
        torch.distributed.gather(tensor=_tensor, gather_list=_tensor_list, async_op=False)
    else:
        torch.distributed.gather(tensor=_tensor, dst=0, async_op=True)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        for t1, t2 in zip(_tensor_ref, _tensor_list):
            assert torch.equal(t1.cpu(), t2.cpu()) and t1.dtype == dtype and t1.dtype == t2.dtype


def main(hpu, world_size, dtype, func):
    _ = DistSetup(hpu, world_size)
    pr = cProfile.Profile()
    pr.enable()
    func(hpu, world_size, dtype)
    pr.disable()
    if hpu == 0:
        stats = pstats.Stats(pr).sort_stats("tottime")
        stats.print_stats()


@pytest.mark.timeout(100)
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("n_hpus", [8])
@pytest.mark.parametrize(
    "func",
    [
        sync_reduce_func,
        sync_reduceScatter_func,
        sync_allReduce_func,
        sync_allReduce_tensorLife,
        async_allReduce_tensorLife,
        sync_allGather_func,
        sync_allGather_func_uneven_size,
        async_allGather_func_uneven_size,
        sync_gather_func,
        sync_broadcast_func,
        sync_broadcast_tensorLife,
        async_broadcast_tensorLife,
        # barrier_func,
        sendReceive_func,
    ],
)
def test_first(func, n_hpus, dtype):
    mp.spawn(main, args=(n_hpus, dtype, func), nprocs=n_hpus, join=True)

    assert True


@pytest.mark.timeout(100)
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("n_hpus", [8])
@pytest.mark.parametrize(
    "func",
    [
        sync_allReduce_func,
        computeAllreduceCompute_perf,
    ],
)
def test_second(func, n_hpus, dtype):
    mp.spawn(main, args=(n_hpus, dtype, func), nprocs=n_hpus, join=True)

    assert True


@pytest.mark.timeout(300)
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("n_hpus", [8])
@pytest.mark.parametrize("func", [hostsync_broadcast])
def test_third(func, n_hpus, dtype):
    mp.spawn(main, args=(n_hpus, dtype, func), nprocs=n_hpus, join=True)

    assert True


@pytest.mark.timeout(300)
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("n_hpus", [2])
@pytest.mark.parametrize("func", [sync_gather_func])
def test_fourth(func, n_hpus, dtype):
    mp.spawn(main, args=(n_hpus, dtype, func), nprocs=n_hpus, join=True)

    assert True
