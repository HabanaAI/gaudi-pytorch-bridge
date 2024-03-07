###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

# Env Flags: PT_ENABLE_INTER_HOST_CACHING=1 PT_RECIPE_CACHE_PATH="/tmp/MyCache" LOG_LEVEL_HOSTSTAT=0 PT_HPU_ENABLE_EXECUTION_THREAD=0 PT_CACHE_FOLDER_SIZE_MB=1
# Pytest flags: --capture=fd --log-cli-level=INFO

import logging
import os

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI
from test_utils import hpu as device

ITER = 5
GAUDI_PER_HLS = 8


def distSetup(rank, world_size):
    dist._DEFAULT_FIRST_BUCKET_BYTES = 500 * 1024 * 1024
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def distCleanup(rank):
    dist.destroy_process_group()


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.L1 = torch.nn.Conv2d(3, 1, kernel_size=7)

    def forward(self, x):
        out = torch.sigmoid(self.L1(x))
        return out


@pytest.fixture()
def rank(capfd, caplog):
    caplog.set_level(logging.ERROR)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    yield rank
    captured = capfd.readouterr()
    outLines = captured.err.splitlines()

    logger = logging.getLogger("Test Logs")
    with caplog.at_level(logging.INFO):
        for line in outLines:
            if "CACHEFILE" in line or "INTERHOST" in line:
                logger.info(line)


@pytest.fixture
def world_size():
    comm = MPI.COMM_WORLD
    return comm.Get_size()


@pytest.fixture
def set_env(rank, world_size):
    os.environ["RANK"] = str(rank)
    local_rank_s = str(rank % GAUDI_PER_HLS)
    # Bridge is using "HLS_MODULE_ID", but "ID" is still needed for internal synapse logging.
    os.environ["ID"] = str(rank)
    os.environ["HLS_MODULE_ID"] = local_rank_s
    os.environ["LOCAL_RANK"] = local_rank_s

    distSetup(rank, world_size)
    yield
    distCleanup(rank)


@pytest.fixture
def network(set_env):
    net = NeuralNetwork().to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, bucket_cap_mb=500)
    return net


@pytest.fixture
def optimizer(network):
    opt = torch.optim.SGD(network.parameters(), lr=0.001)
    return opt


# terminate called after throwing an instance of 'c10::Error'
#   what():  Host barrier Key error
# Exception raised from hostBarrier at ../../../../../../repos/pytorch-integration/python_packages/habana_frameworks/torch/distributed/hccl/process_group_hccl_base.cpp:1033 (most recent call first):
# frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x6c (0x7f80cab2f53c in /home/jenkins/6bfb6a4b/workspace/pytorch_modules/Tests/Test_pytorch_modules_gaudi_sim_cpp_suite_master_next/.venv/lib/python3.8/site-packages/torch/lib/libc10.so)
# frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, char const*) + 0x84 (0x7f80caaf5220 in /home/jenkins/6bfb6a4b/workspace/pytorch_modules/Tests/Test_pytorch_modules_gaudi_sim_cpp_suite_master_next/.venv/lib/python3.8/site-packages/torch/lib/libc10.so)
# frame #2: c10d::ProcessGroupHcclBase::hostBarrier() + 0x3bf (0x7f80afaf1cff in /home/jenkins/6bfb6a4b/workspace/pytorch_modules/Tests/Test_pytorch_modules_gaudi_sim_cpp_suite_master_next/.venv/lib/python3.8/site-packages/habana_frameworks/torch/distributed/_hccl_C.so)
# frame #3: c10d::ProcessGroupHCCL::destroy() + 0x28 (0x7f80afb17988 in /home/jenkins/6bfb6a4b/workspace/pytorch_modules/Tests/Test_pytorch_modules_gaudi_sim_cpp_suite_master_next/.venv/lib/python3.8/site-packages/habana_frameworks/torch/distributed/_hccl_C.so)
# frame #4: c10d::ProcessGroupHCCL::~ProcessGroupHCCL() + 0x23e (0x7f80afb17e0e in /home/jenkins/6bfb6a4b/workspace/pytorch_modules/Tests/Test_pytorch_modules_gaudi_sim_cpp_suite_master_next/.venv/lib/python3.8/site-packages/habana_frameworks/torch/distributed/_hccl_C.so)
# frame #5: c10d::ProcessGroupHCCL::~ProcessGroupHCCL() + 0xd (0x7f80afb17e9d in /home/jenkins/6bfb6a4b/workspace/pytorch_modules/Tests/Test_pytorch_modules_gaudi_sim_cpp_suite_master_next/.venv/lib/python3.8/site-packages/habana_frameworks/torch/distributed/_hccl_C.so)
# frame #6: c10d::Reducer::~Reducer() + 0x448 (0x7f80cfe8f0e8 in /home/jenkins/6bfb6a4b/workspace/pytorch_modules/Tests/Test_pytorch_modules_gaudi_sim_cpp_suite_master_next/.venv/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so)
@pytest.mark.skip(reason="may crash device during teardown")
class TestRemoteCache:
    # Rank 0 compiles and others reuse
    def test_zero_to_all(rank, world_size, network, optimizer):

        if rank == 0:
            dim = 100
        else:
            dim = 99

        iter = ITER * world_size
        for _ in np.arange(iter):
            inp = torch.ones(1, 3, dim, dim).to(device)
            out = network(inp)
            optimizer.zero_grad()
            loss = out.sum()
            loss.backward()
            optimizer.step()
            htcore.mark_step()

            dim = dim + 1

    # Rank 8 compiles and others reuse
    def test_eight_to_all(rank, world_size, network, optimizer):

        if rank == 8:
            dim = 100
        else:
            dim = 99

        iter = ITER * world_size
        for _ in np.arange(iter):
            inp = torch.ones(1, 3, dim, dim).to(device)
            out = network(inp)
            optimizer.zero_grad()
            loss = out.sum()
            loss.backward()
            optimizer.step()
            htcore.mark_step()

            dim = dim + 1

    # Each rank compiles a unique recipe first and then everyone shares
    def test_per_rank_unique_compile(rank, world_size, network, optimizer):

        for j in np.arange(2):
            if j == 0:
                dim = 99 + (rank * ITER)
                iter = ITER + 1
            else:
                dim = 100
                iter = ITER * world_size

            for _ in np.arange(iter):
                inp = torch.ones(1, 3, dim, dim).to(device)
                out = network(inp)
                optimizer.zero_grad()
                loss = out.sum()
                loss.backward()
                optimizer.step()
                htcore.mark_step()

                dim = dim + 1

    # Basic eviction test that assumes small PT_CACHE_FOLDER_SIZE_MB
    # i.e. PT_CACHE_FOLDER_SIZE_MB=1
    def test_eviction_basic(rank, world_size, network, optimizer):
        _ITER = 50
        dim = 99 + (rank * _ITER)
        for _ in np.arange(_ITER):
            inp = torch.ones(1, 3, dim, dim).to(device)
            _ = network(inp)
            htcore.mark_step()

            dim = dim + 1

        files_count = len(list(os.scandir(os.environ["PT_RECIPE_CACHE_PATH"])))
        assert files_count < world_size * _ITER * 2
