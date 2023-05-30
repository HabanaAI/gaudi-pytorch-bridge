import os
import torch
from typing import Tuple

from habana_frameworks.torch.hpu import HABANA_VISIBLE_MODULES_VAR, HLS_MODULE_ID_VAR
from habana_frameworks.torch.utils.experimental.distributed_emulation import \
    distributed_emulation_apply_if_enabled,\
    is_distributed_emulation_enabled

_lazy_mode = int(os.environ.get("PT_HPU_LAZY_MODE", "1"))
_lazy_collectives_enabled = os.environ.get("PT_HPU_ENABLE_LAZY_COLLECTIVES", "False").lower() in ["true", "1"]
if _lazy_mode == 0:
    # PT 2.0 eager mode
    from habana_frameworks.torch.distributed._hccl_eager_C import *
elif _lazy_collectives_enabled:
    # Lazy mode with lazy collectives
    from habana_frameworks.torch.distributed._hccl_lazy_C import *
else:
    # Lazy mode without lazy collectives
    from habana_frameworks.torch.distributed._hccl_C import *


distributed_emulation_apply_if_enabled()


def _setup_module_id(local_rank=-1, world_size=1):

    if HLS_MODULE_ID_VAR in os.environ.keys():
        # Module id already set, exiting.
        return

    if local_rank == -1 or world_size == 1 or is_distributed_emulation_enabled():
        # In case local rank is not available in env we do net set HLS_MODULE_ID
        # PT_BRIDGE will acquire device by type.
        # This would also apply in single node (or emulation) scenarios as there is
        # no benefit for using specific card.
        return

    if HABANA_VISIBLE_MODULES_VAR in os.environ.keys():
        visible_modules = os.environ[HABANA_VISIBLE_MODULES_VAR].split(",")
        assert local_rank < len(visible_modules), f"""There is not enough devices
        available for training. Please verify if {HABANA_VISIBLE_MODULES_VAR}
        is set correctly."""
        os.environ[HLS_MODULE_ID_VAR] = visible_modules[local_rank]
        return
    # In all other cases strict mapping of local_rank -> module_id allows easier NUMA or MPI binding.
    os.environ[HLS_MODULE_ID_VAR] = str(local_rank)


def _setup_user_overrides(world_size=None, rank=None, local_rank=None):
    # Handle override provided by user (if any):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)


def _setup_environment_from_mpi():
    OMPI_VARIABLES_MAPPING = {
        'OMPI_COMM_WORLD_LOCAL_RANK': 'LOCAL_RANK',
        'OMPI_COMM_WORLD_SIZE': 'WORLD_SIZE',
        'OMPI_COMM_WORLD_RANK': 'RANK'
    }

    if all(key in os.environ.keys() for key in OMPI_VARIABLES_MAPPING.values()):
        # All environment variables are already set. We will not override it.
        return

    if all(key in os.environ.keys() for key in OMPI_VARIABLES_MAPPING.keys()):
        for mpi_env_var_name in OMPI_VARIABLES_MAPPING.keys():
            env_var_name = OMPI_VARIABLES_MAPPING[mpi_env_var_name]
            os.environ[env_var_name] = os.environ[mpi_env_var_name]

    # This generally should be set outside but in case they are not,
    # we at least be still able to run in single node (ScaleUp) scenarios
    if os.getenv('MASTER_ADDR') is None:
        os.environ['MASTER_ADDR'] = "localhost"
    if os.getenv('MASTER_PORT') is None:
        os.environ['MASTER_PORT'] = "12345"


def _read_values_from_env():
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    return world_size, rank, local_rank


def initialize_distributed_hpu(world_size=None, rank=None, local_rank=None) -> Tuple[int, int, int]:
    r"""Initializes and returns distributed configuration
    Returns world_size, rank and local_rank if the processes
    are launched using either MPI or torchrun related APIS
    """

    if all(v is not None for v in [world_size, rank, local_rank]):
        _setup_user_overrides(world_size, rank, local_rank)
    else:
        _setup_environment_from_mpi()

    world_size, rank, local_rank = _read_values_from_env()

    _setup_module_id(local_rank=local_rank, world_size=world_size)

    # setup id for synapse logging
    if rank != -1:
        os.environ["ID"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    return world_size, rank, local_rank


initialize_distributed_hpu()


def _create_process_group_hccl(store, rank, size, timeout):
    return ProcessGroupHCCL(
        store,
        rank,
        size,
        timeout)


torch.distributed.Backend.register_backend("hccl", _create_process_group_hccl)
