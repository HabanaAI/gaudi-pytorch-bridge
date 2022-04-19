import os
import torch
from habana_frameworks.torch.distributed._hccl_C import *

def initialize_distributed_hpu() -> None:
    r"""Initializes and returns distributed configuration
    Returns world_size, rank and local_rank if the processes
    are launched using either MPI or torchrun related APIS
    """
    world_size = 1
    rank = -1
    local_rank = -1
    if ('WORLD_SIZE' in os.environ and
        'RANK' in os.environ and
        'local_rank' in os.environ):
        world_size = int(os.environ["WORLD_SIZE"])
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif ('OMPI_COMM_WORLD_LOCAL_RANK' in os.environ and
          'OMPI_COMM_WORLD_SIZE' in os.environ and
          'OMPI_COMM_WORLD_RANK' in os.environ):
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        rank       = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        try:
            global mpi_comm
            from mpi4py import MPI
            mpi_comm = MPI.COMM_WORLD
            world_size = mpi_comm.Get_size()
            if world_size > 1:
                rank = mpi_comm.Get_rank()
                local_rank = rank
            else:
                raise("Single MPI process")
        except Exception as e:
            pass

    if world_size > 1 and local_rank != -1:
        os.environ["ID"] = str(local_rank)
    return world_size, rank, local_rank
