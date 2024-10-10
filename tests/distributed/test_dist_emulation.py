import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from pytest_working.test_utils import env_var_in_scope
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP


def _zero1_with_ddp_worker(rank, world_size):
    os.environ["RANK"] = str(rank)

    import habana_frameworks.torch.core as htcore

    device = torch.device("hpu")

    torch.manual_seed(rank)
    input = torch.randn(2000, 2000)
    label = torch.randn(2000, 2000)

    # create default process group
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    # create local model
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(device) for _ in range(20)])
    model = model.to(device)

    # construct DDP model
    ddp_model = DDP(copy.deepcopy(model).to(device), bucket_cap_mb=10000 * 1024 * 1024, gradient_as_bucket_view=True)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()

    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=0.01,
        overlap_with_ddp=False,
        weight_decay=1e-2,
        eps=1e-8,
    )

    for i in range(10):
        # forward pass
        outputs = ddp_model(input.to(device))
        labels = label.to(device)
        # backward pass
        loss = loss_fn(outputs, labels)
        loss.backward()
        # break the graph
        htcore.mark_step()
        # update parameters

        optimizer.step()
        htcore.mark_step()

        if rank == 0:
            print(" loss for step ", i, " is ", loss.to("cpu"))

    print(f"[rank={rank}] params sum is: {sum(model.parameters()).sum()}")


def test_distributed_single_rank_emulation():
    working_rank = 0
    world_size = 8

    env_vars = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        "PT_HPU_EMULATE_DISTRIBUTED": 1,
        "PT_HPU_EMULATE_DISTRIBUTED_SINGLE_RANK": working_rank,
    }

    with env_var_in_scope(env_vars):
        mp.spawn(_zero1_with_ddp_worker, args=(world_size,), nprocs=world_size, join=True)


def test_distributed_emulation_skip_collective_only():
    env_vars = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        "PT_HPU_EMULATE_DISTRIBUTED": 1,
    }

    world_size = 8

    with env_var_in_scope(env_vars):
        mp.spawn(_zero1_with_ddp_worker, args=(world_size,), nprocs=world_size, join=True)
