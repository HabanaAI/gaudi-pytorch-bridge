import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch.manual_seed(0)
device = torch.device("hpu")


class DistSetup:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12340"
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        # Following Code is to ensure that HCL_Init is done
        _tensor = torch.ones(1).to(device)
        torch.distributed.all_reduce(_tensor)

    def __del__(self):
        dist.destroy_process_group()


def reduce_op_worker(rank, world_size):
    _ = DistSetup(rank, world_size)

    if rank == 0:
        tensor0 = torch.tensor([0.0, 0.0, 0.0]).to("hpu")
    else:
        tensor0 = torch.tensor([-1.75, -1.75, -1.75]).to("hpu")

    tensor = tensor0[1:3:1].detach()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        tensor_ref = torch.tensor([0.0000, -1.7500, -1.7500])
        assert torch.equal(tensor0, tensor_ref)


if __name__ == "__main__":
    world_size = 2

    mp.spawn(reduce_op_worker, args=(world_size,), nprocs=world_size, join=True)
    os.environ["PT_HPU_LAZY_MODE"] = "2"
    mp.spawn(reduce_op_worker, args=(world_size,), nprocs=world_size, join=True)
    os.environ["PT_HPU_LAZY_MODE"] = "1"
