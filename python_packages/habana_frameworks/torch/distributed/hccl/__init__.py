import os
import torch
from habana_frameworks.torch.distributed._hccl_C import *

def checkVisibleDevices(rank):
    HABANA_VISIBLE_MODULES_VAR = "HABANA_VISIBLE_MODULES"
    HABANA_VISIBLE_DEVICES_VAR = "HABANA_VISIBLE_DEVICES"
    HABANA_DEVICE_ID_VAR = "ID"

    if HABANA_VISIBLE_MODULES_VAR in os.environ.keys():
        visible_modules = os.environ[HABANA_VISIBLE_MODULES_VAR].split(",")
        assert rank < len(visible_modules), f"""There is not enough devices
        available for training. Please verify if {HABANA_VISIBLE_MODULES_VAR}
        is set correctly."""
        os.environ[HABANA_DEVICE_ID_VAR] = visible_modules[rank]
        return
    elif HABANA_VISIBLE_DEVICES_VAR in os.environ.keys():
        visible_modules = os.environ[HABANA_VISIBLE_DEVICES_VAR].split(",")
        assert rank < len(visible_modules), f"""There is not enough devices
        available for training. Please verify if {HABANA_VISIBLE_DEVICES_VAR}
        is set correctly."""
        os.environ[HABANA_DEVICE_ID_VAR] = visible_modules[rank]
        return      

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
        'LOCAL_RANK' in os.environ):
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
        checkVisibleDevices(local_rank)
    return world_size, rank, local_rank
