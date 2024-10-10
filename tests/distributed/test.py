import faulthandler
import sys
import traceback

import torch
import torch.distributed as dist

faulthandler.enable(all_threads=True)
import os

from habana_frameworks.torch.utils.library_loader import load_habana_module

load_habana_module()

os.environ["ID"] = os.getenv("RANK")
rank = int(os.getenv("RANK"))
world_size = int(os.getenv("WORLD_SIZE"))

a = torch.tensor([1.0, 2.0]).to("hpu")

try:
    global_comm = dist.init_process_group("hccl")
except:
    print("Exception during load")
    traceback.print_exc()
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)
    exit(1)

print("Allreduce")
a = torch.tensor([1.0, 2.0]).to("hpu")
dist.all_reduce(a)
print(a.to("cpu"))

print("Broadcast")
a = torch.tensor([1.0, 2, 0, 3.0, float(rank)]).to("hpu")
dist.broadcast(a, 0)  # broadcast from rank 0
print(a.to("cpu"))

print("Allgather")
op_tensor_list = [torch.empty(2, device="hpu") for i in range(world_size)]
a = torch.tensor([1.0, 2.0]).to("hpu")
dist.all_gather(op_tensor_list, a)

print("AlltoAll")
ip_tensor = torch.tensor([1.0, 2.0]).to("hpu")
op_tensor = torch.empty(2, device="hpu")
dist.all_to_all_single(op_tensor, ip_tensor)
print(op_tensor.to("cpu"))


print("Allreduce integer")
a = torch.tensor([1, 2]).to("hpu")
dist.all_reduce(a)
print(a.to("cpu"))
print(a.dtype)

import habana_frameworks.torch.core as htcore

print(htcore.get_device_name(rank))
# Command to run the test
# HCL_CONFIG_PATH=hls1.json python -um torch.distributed.launch --nproc_per_node=8 --use_env test.py
