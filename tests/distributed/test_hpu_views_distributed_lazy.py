import os

import habana_frameworks.torch.core as htcore
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch.manual_seed(0)
device = torch.device("hpu")

os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "1"


class DistSetup:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12340"
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    def __del__(self):
        dist.destroy_process_group()


def reduce_op_worker(rank, world_size):
    _ = DistSetup(rank, world_size)

    # Maximum value along vocab dimension across all GPUs.
    vocab_parallel_logits = torch.randn([1, 4, 8]).to("hpu")
    target = torch.zeros([1, 4]).to(torch.int64).to("hpu")

    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    # Subtract the maximum value.
    htcore.mark_step()
    vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

    # Get the partition's vocab indecies
    partition_vocab_size = vocab_parallel_logits.size()[-1]

    masked_target = target

    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)

    # # All reduce is needed to get the chunks from other GPUs.
    torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, async_op=True)

    # Sum of exponential of logits along vocab dimension across all GPUs.
    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)

    htcore.mark_step()

    tensor_ref = torch.tensor([3.1030, 3.7284, 2.1220, 3.4019])
    assert torch.allclose(sum_exp_logits.cpu(), tensor_ref, atol=0.001, rtol=0.001)


if __name__ == "__main__":
    world_size = 2
    os.environ["PT_HPU_LAZY_MODE"] = "2"
    mp.spawn(reduce_op_worker, args=(world_size,), nprocs=world_size, join=True)

    os.environ["PT_HPU_LAZY_MODE"] = "1"
    mp.spawn(reduce_op_worker, args=(world_size,), nprocs=world_size, join=True)
