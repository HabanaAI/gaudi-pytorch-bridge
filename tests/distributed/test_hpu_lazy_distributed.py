import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("hpu")


class DistSetup:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12340"
        os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"
        os.environ["PT_HPU_ENABLE_SFG"] = "1"
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        # Following Code is to ensure that HCL_Init is done
        _tensor = torch.ones(1).to(device)
        torch.distributed.all_reduce(_tensor)

    def __del__(self):
        dist.destroy_process_group()


def all_reduce_op_worker(rank, world_size):
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


def reduce_op_worker(rank, world_size):
    _ = DistSetup(rank, world_size)

    if rank == 0:
        tensor0 = torch.tensor([0.0, 0.0, 0.0]).to("hpu")
    else:
        tensor0 = torch.tensor([-1.75, -1.75, -1.75]).to("hpu")

    tensor = tensor0[1:3:1].detach()

    dist.reduce(tensor, 0)

    if rank == 0:
        tensor_ref = torch.tensor([0.0000, -1.7500, -1.7500])
        assert torch.equal(tensor0, tensor_ref)


def init_weights(m):
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)


class SplitBatchLinear(nn.Module):
    def __init__(self, weight_size, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(weight_size, dtype=dtype))
        self.num_shards = 4
        self.output_shape = weight_size[1]
        self.bias = nn.Parameter(torch.empty(1, weight_size[1], dtype=dtype))
        init_weights(self)

    def forward(self, inputs):
        shard_size = inputs.size()[0] // self.num_shards
        start_offset = 0
        output_size = list(inputs.size())
        output_size[-1] = self.output_shape
        output = torch.empty(output_size, dtype=inputs.dtype, layout=inputs.layout, device=inputs.device)
        works = []
        output_shards = []
        for i in range(self.num_shards):
            curr_shard_size = shard_size if i < self.num_shards - 1 else inputs.size()[0] - start_offset
            output_shard = torch.matmul(inputs[start_offset : start_offset + shard_size], self.weight)
            work = torch.distributed.all_reduce(output_shard, async_op=True)
            works.append(work)
            output_shards.append(output_shard)
            start_offset = start_offset + curr_shard_size

        for work in works:
            work.wait()

        output = torch.cat(output_shards, dim=0)
        return output + self.bias


class ToyModel(nn.Module):
    def __init__(self, weight_size, dtype, world_size):
        super().__init__()
        self.split_batch_layer1 = SplitBatchLinear(weight_size, dtype)
        self.split_batch_layer2 = SplitBatchLinear(weight_size, dtype)

    def forward(self, inputs):
        batch_layer1_out = self.split_batch_layer1(inputs)
        batch_layer2_out = self.split_batch_layer2(batch_layer1_out)
        return batch_layer2_out


def split_batch_linear_with_cat(rank, world_size):
    os.environ["PT_HPU_COMPILE_USE_RECIPES"] = "1"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["PT_HPU_ENABLE_SFG"] = "1"
    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"
    os.environ["PT_HPU_DISABLE_ASYNC_COLLECTIVE"] = "1"
    torch.manual_seed(12345)
    comm_group = torch.distributed.init_process_group("hccl")
    device = torch.device("hpu")
    hidden_dimention = 14 * 1024
    model = ToyModel([hidden_dimention, hidden_dimention], torch.bfloat16, world_size)
    model.to(device)
    iter_10_out_ref = torch.tensor(-7392.0, dtype=torch.bfloat16, device="cpu")

    def run_iterations(prof=None):
        out = 0
        for cnt in range(10):
            inp = torch.randn([60, hidden_dimention], dtype=torch.bfloat16, device=device)
            with torch.no_grad():
                output = model(inp)
            out = output.to("cpu").sum()
            # print("RANK:", os.getenv("RANK"), " Output ", out)
        return out

    out = run_iterations()
    assert torch.equal(iter_10_out_ref, out)


if __name__ == "__main__":
    world_size = 2
    mp.spawn(all_reduce_op_worker, args=(world_size,), nprocs=world_size, join=True)
    mp.spawn(reduce_op_worker, args=(world_size,), nprocs=world_size, join=True)
    mp.spawn(split_batch_linear_with_cat, args=(world_size,), nprocs=world_size, join=True)
