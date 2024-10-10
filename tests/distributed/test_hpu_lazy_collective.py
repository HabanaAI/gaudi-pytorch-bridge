###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os

import habana_frameworks.torch as ht
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# This is simplified UT based on SW-190459


class DistSetup:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12340"
        os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    def __del__(self):
        dist.destroy_process_group()


def all_reduce_op_worker(rank, world_size):
    _ = DistSetup(rank, world_size)

    # embedding = nn.Embedding(10, 1).to('hpu')
    # input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).to('hpu')
    # mask = torch.ones(2, 4, device='hpu', dtype=torch.bool)
    # unmask = mask.unsqueeze(dim=2)
    # out = embedding(input)
    # out.masked_fill_(unmask, 1)
    # dist.all_reduce(out, op=dist.ReduceOp.SUM)

    unmask = torch.ones(2, 4, 1, device="hpu", dtype=torch.bool)
    input = torch.empty(2, 4, 1, device="hpu")
    out = input.relu()
    out.masked_fill_(unmask, rank)
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    if rank == 0:
        out_ref = torch.ones(2, 4, 1, device="cpu")
        assert torch.allclose(out.to("cpu"), out_ref)


def test_lazy_allreduce():
    world_size = 2
    mp.spawn(all_reduce_op_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    test_lazy_allreduce()
