# ******************************************************************************
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
# ******************************************************************************
import os
import sys
from functools import partial

"""
Distributed emulation simulates running multinode workload on single Gaudi
device.

Distributed emulation can be enabled via setting PT_HPU_EMULATE_DISTRIBUTED env
variable to "1" or "True". When this flag is set then collective communication
is skipped. Additionally when PT_HPU_EMULATE_DISTRIBUTED_SINGLE_RANK is set to
specific rank, then this rank will be kept alive and others will gracefully exit.

In order to use distributed emulation habana_frameworks.torch.distributed.hccl
has to be imported before opening the hpu device and initializing process group.
"""


def _distributed_emulation_patch_store_barrier():
    """
    Function monkey patches the store_based_barrier and constructor of TCPStore
    which are executed during process group initialization to synchronize all
    workers. In distributed emulation, where only one worker is actually
    executed (other are killed) the synchronization needs to be skipped.
    """
    import torch.distributed.distributed_c10d

    def store_based_barrier_dummy(rank, store, timeout):
        pass

    torch.distributed.distributed_c10d._store_based_barrier = store_based_barrier_dummy

    import torch.distributed.rendezvous
    from torch.distributed import TCPStore as TCPStore_orig

    sys.modules["torch.distributed.rendezvous"].TCPStore = partial(TCPStore_orig, wait_for_workers=False)


def is_distributed_emulation_enabled():
    if os.environ.get("PT_HPU_EMULATE_DISTRIBUTED", "False").lower() in ["true", "1"]:
        return True
    return False


def distributed_emulation_apply_if_enabled():
    hpu_emulate_distributed_single_rank = os.environ.get("PT_HPU_EMULATE_DISTRIBUTED_SINGLE_RANK", None)
    if hpu_emulate_distributed_single_rank:
        hpu_emulate_distributed_single_rank = int(hpu_emulate_distributed_single_rank)

    hpu_emulate_distributed = is_distributed_emulation_enabled()

    if hpu_emulate_distributed and hpu_emulate_distributed_single_rank is not None:
        _distributed_emulation_patch_store_barrier()

        try:
            rank = int(os.environ["RANK"])
        except KeyError:
            print("In order to use distributed emulation the 'RANK' env var has to be set to worker rank")
            raise

        if hpu_emulate_distributed_single_rank != rank:
            print(f"Single rank emulation mode, rank {rank} terminating")
            exit(0)
