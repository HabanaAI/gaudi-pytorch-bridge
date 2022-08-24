# Env Flags: PT_ENABLE_INTER_HOST_CACHING=1 PT_RECIPE_CACHE_PATH="/tmp/MyCache" PT_HPU_LOG_MOD_MASK=0x1000 PT_HPU_LOG_TYPE_MASK=FF PT_HPU_ENABLE_EXECUTION_THREAD=0 PT_CACHE_FOLDER_SIZE_MB=1
# Pytest flags: --capture=fd --log-cli-level=INFO

import os
import pytest
import logging

import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug

from mpi4py import MPI
from habana_frameworks.torch.utils.library_loader import load_habana_module

os.environ["PT_HPU_LAZY_MODE"] = "1"
ITER = 5
GAUDI_PER_HLS = 8

load_habana_module()
htdebug._enable_weight_permute_pass(True)
device = torch.device('hpu')

def distSetup(rank, world_size):
  dist._DEFAULT_FIRST_BUCKET_BYTES = 500 * 1024 * 1024
  import habana_frameworks.torch.distributed.hccl
  dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

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

  logger = logging.getLogger('Test Logs')
  with caplog.at_level(logging.INFO):
    for line in outLines:
      if 'CACHEFILE' in line or 'INTERHOST' in line:
        logger.info(line)

@pytest.fixture
def world_size():
  comm = MPI.COMM_WORLD
  return comm.Get_size()

@pytest.fixture
def set_env(rank, world_size):
  os.environ["ID"] = str(rank % GAUDI_PER_HLS)
  os.environ["RANK"] = str(rank)
  os.environ["LOCAL_RANK"] = str(rank % GAUDI_PER_HLS)

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

# Rank 0 compiles and others reuse
def test_zero_to_all(rank, world_size, network, optimizer):

  if (rank == 0):
    dim = 100
  else:
    dim = 99

  iter = ITER * world_size
  for i in np.arange(iter):

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

  if (rank == 8):
    dim = 100
  else:
    dim = 99

  iter = ITER * world_size
  for i in np.arange(iter):

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

    for i in np.arange(iter):

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

  files_count = len(list(os.scandir(os.environ['PT_RECIPE_CACHE_PATH'])))
  assert(files_count < world_size * _ITER * 2)
  