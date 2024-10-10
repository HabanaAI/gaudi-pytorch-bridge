import faulthandler
import sys
import traceback

import torch
import torch.distributed as dist

faulthandler.enable(all_threads=True)
import contextlib
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
    a = torch.tensor([2.0], dtype=torch.float).to("hpu")
    torch.distributed.all_reduce(a)
except:
    print("Exception")
    traceback.print_exc()
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)


def _build_tensor(size, value=None):
    if value is None:
        value = size
    return torch.empty(size, dtype=torch.float).fill_(value).to("hpu")


@contextlib.contextmanager
def _batch_p2p_hccl_manager():
    if torch.distributed.get_backend() == "hccl":
        torch.distributed.ProcessGroupHCCL._group_start()
    try:
        yield
    finally:
        if torch.distributed.get_backend() == "hccl":
            torch.distributed.ProcessGroupHCCL._group_end()


def test_batch_isend_irecv():
    p2p_op_list = []
    for src in range(0, dist.get_world_size()):
        send_tensor = _build_tensor(rank + 1)
        recv_tensor = _build_tensor(src + 1)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, src)
        p2p_op_list.append(recv_op)
        send_op = dist.P2POp(dist.isend, send_tensor, src)
        p2p_op_list.append(send_op)

    with _batch_p2p_hccl_manager():
        reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()


test_batch_isend_irecv()
# Command to run the test
# python -um torch.distributed.launch --nproc_per_node=8 --use_env test_batch_isend_irecv.py
