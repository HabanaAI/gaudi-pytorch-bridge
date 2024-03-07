# cmd to run : NATIVE=1 OVERLAP=1 python -u testZero1.py
# NATIVE=1 to work with native optimizer
# OVERLAP=1 to enable overlap_with_ddp feature of Zero1

import os
import time

import habana_frameworks.torch.core as htcore
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from habana_frameworks.torch.hpex.optimizers import FusedAdamW

# To register a functional optimizer, import the optimizer and
# invoke register_functional_optim(key,optimizer) from torch.distributed.optim.utils
# to register the optimizer
from habana_frameworks.torch.hpex.optimizers.distributed import FusedAdamW as FunctionalFusedAdamW
from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import hook_with_zero_step_interleaved
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.utils import register_functional_optim
from torch.nn.parallel import DistributedDataParallel as DDP

register_functional_optim(FunctionalFusedAdamW, FunctionalFusedAdamW)

NATIVE = 0
use_native = int(os.environ["NATIVE"])
OVERLAP = 0
use_overlap = int(os.environ["OVERLAP"])
WARMUP_STEPS = 2

device = torch.device("hpu")
torch.manual_seed(0)
input = [torch.randn(2000, 2000), torch.randn(2000, 2000)]
label = [torch.randn(2000, 2000), torch.randn(2000, 2000)]


def example(rank, world_size, use_zero):
    torch.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # create default process group
    dist.init_process_group("hccl", rank=rank, world_size=world_size)

    # create local model
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(device) for _ in range(20)])
    model = model.to(device)
    # construct DDP model
    import copy

    ddp_model = DDP(copy.deepcopy(model).to(device), bucket_cap_mb=10000 * 1024 * 1024, gradient_as_bucket_view=True)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=torch.optim.Adam if use_native else FunctionalFusedAdamW,
            lr=0.01,
            # parameters_as_bucket_view=True,
            overlap_with_ddp=True if use_overlap else False,
            weight_decay=1e-2,
            eps=1e-8,
        )
        if use_overlap:
            print("registering comm hook")
            ddp_model.register_comm_hook(
                None, hook_with_zero_step_interleaved(allreduce_hook, ddp_model, optimizer, shard_buckets=True)
            )
    else:
        optimizer = (
            torch.optim.Adam(ddp_model.parameters(), lr=0.01, weight_decay=1e-2, eps=1e-8)
            if use_native
            else FusedAdamW(ddp_model.parameters(), lr=0.01, weight_decay=1e-2, eps=1e-8)
        )
    i = 1
    while i != 10:
        # forward pass
        outputs = ddp_model(input[rank].to(device))
        labels = label[rank].to(device)
        # backward pass
        loss = loss_fn(outputs, labels)
        loss.backward()
        # break the graph
        htcore.mark_step()
        # update parameters
        if ((not use_zero) or (use_zero and not use_overlap)) and (i > WARMUP_STEPS):  # 2 warm-up steps
            optimizer.step()
            htcore.mark_step()
        if rank == 0:
            print(" loss for step ", i, " is ", loss.to("cpu"))
        i = i + 1

    print(f"params sum is: {sum(model.parameters()).sum()}")


def main():
    world_size = 2
    print("=== Using ZeroRedundancyOptimizer ===")
    start_time = time.time()
    mp.spawn(example, args=(world_size, True), nprocs=world_size, join=True)
    print("Time : ", time.time() - start_time)
    print("=== Not Using ZeroRedundancyOptimizer ===")
    start_time = time.time()
    mp.spawn(example, args=(world_size, False), nprocs=world_size, join=True)
    print("Time : ", time.time() - start_time)


if __name__ == "__main__":
    main()
