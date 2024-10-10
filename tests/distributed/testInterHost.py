import os
import random
import time

import habana_frameworks.torch.core as htcore
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from habana_frameworks.torch.utils.library_loader import load_habana_module
from mpi4py import MPI

os.environ["PT_HPU_LAZY_MODE"] = "1"
ITER = 100

torch.manual_seed(0)
load_habana_module()
device = torch.device("hpu")


def setup(rank, world_size):
    print("Setup HCCL")
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    time.sleep(15)


def cleanup(rank):
    print("Cleanup ", rank)
    dist.destroy_process_group()


class NeuralNetwork(torch.nn.Module):
    def __init__(self, val):
        super(NeuralNetwork, self).__init__()
        self.L1 = torch.nn.Linear(10, val)
        self.L2 = torch.nn.Linear(val, 20)
        self.L3 = torch.nn.Linear(20, 2)

    def forward(self, x):
        out = F.relu(self.L1(x))
        out = F.relu(self.L2(out))
        out = torch.sigmoid(self.L3(out))
        return out


def main_worker(hpu, world_size):
    if world_size > 1:
        setup(hpu, world_size)

    random.seed(hpu)

    for j in np.arange(2):
        for i in np.arange(ITER):
            dist.barrier()
            print("Barrier: ", hpu)

            if hpu < 8:
                val = 100 + hpu + (8 * i)
            else:
                val = 100 + 8 * (ITER - 1 - i) + (hpu - 8)
            print(hpu, i, val)
            net = NeuralNetwork(val).to(device)
            inp = torch.ones(1, 10).to(device)

            loss_fn = torch.nn.MSELoss()
            ref = torch.ones(1, 2).to(device)
            opt = torch.optim.SGD(net.parameters(), lr=0.001)

            out = net(inp)
            loss_fn(out, ref).backward()
            opt.step()
            htcore.mark_step()

    if world_size > 1:
        cleanup(hpu)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    wsize = comm.Get_size()
    print(rank, wsize)
    main_worker(rank, wsize)
