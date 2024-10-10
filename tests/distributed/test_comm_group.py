import faulthandler
import sys
import traceback

import torch
import torch.distributed as dist

faulthandler.enable(all_threads=True)
import os

from habana_frameworks.torch.utils.library_loader import load_habana_module

load_habana_module()

if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
elif "OMPI_COMM_WORLD_RANK" in os.environ and "OMPI_COMM_WORLD_SIZE" in os.environ:
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
else:
    sys.exit("Not a multinode run")

os.environ["ID"] = str(rank)

try:
    global_comm = dist.init_process_group("hccl")
except:
    print("Exception")
    traceback.print_exc()
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)


def check_comm_allreduce(rank, comm_ranks, comm_group):
    if rank in comm_ranks:
        reduce_tensor = torch.tensor([rank], dtype=torch.float32).to("hpu")
        dist.all_reduce(reduce_tensor, group=comm_group)
        reduce_sum = sum(comm_ranks)
        if reduce_tensor.item() != reduce_sum:
            print("Comm group reduction failed for ranks ", comm_ranks)


def check_comm_broadcast(rank, comm_ranks, comm_group):
    if rank in comm_ranks:
        # Check broadcast from rank 0
        broadcast_tensor = torch.tensor([rank], dtype=torch.float32).to("hpu")
        dist.broadcast(broadcast_tensor, comm_ranks[0], group=comm_group)
        if broadcast_tensor.item() != comm_ranks[0]:
            print("Comm group broadcast failed")
        # Check broadcast from different rank
        broadcast_tensor = torch.tensor([rank], dtype=torch.float32).to("hpu")
        dist.broadcast(broadcast_tensor, comm_ranks[-1], group=comm_group)
        if broadcast_tensor.item() != comm_ranks[-1]:
            print("Comm group broadcast failed")


def check_comm_allgather(rank, comm_ranks, comm_group):
    if rank in comm_ranks:
        input_gather_tensor = torch.tensor([rank], dtype=torch.float32).to("hpu")
        op_tensor_list = [torch.empty_like(input_gather_tensor) for rank in comm_ranks]
        dist.all_gather(op_tensor_list, input_gather_tensor, group=comm_group)
        for pos, rank in enumerate(comm_ranks):
            if op_tensor_list[pos].item() != rank:
                print("Comm group allgather failed")


def check_comm_collectives(rank, comm_ranks, comm_group=None):
    if comm_group is None:
        comm_group = dist.new_group(ranks=comm_ranks)
    check_comm_allreduce(rank, comm_ranks, comm_group)
    check_comm_broadcast(rank, comm_ranks, comm_group)
    check_comm_allgather(rank, comm_ranks, comm_group)


global_comm_ranks = list(range(world_size))
check_comm_collectives(rank, global_comm_ranks, global_comm)
# Check subgroup of first half ranks
check_comm_collectives(rank, global_comm_ranks[: len(global_comm_ranks) // 2])
# Check subgroup of sencond half ranks
check_comm_collectives(rank, global_comm_ranks[len(global_comm_ranks) // 2 :])
# Check subgroup of alternate ranks starting from 0
check_comm_collectives(rank, global_comm_ranks[0::2])
# Check subgroup of alternate rnaks starting from 1
check_comm_collectives(rank, global_comm_ranks[1::2])

# Command to run the test
# python -um torch.distributed.launch --nproc_per_node=8 --use_env test_comm_group.py
