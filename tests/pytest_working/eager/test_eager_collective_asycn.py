import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
from torchvision import datasets, transforms
from torch.nn import Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from copy import deepcopy
import habana_frameworks.torch.hpu
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    import habana_frameworks.torch.distributed.hccl
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

device_hpu = torch.device('hpu')
def simple(rank, world_size, args):
    device = f'{device_hpu}'
    setup(rank, world_size)
    # test all_gather
    input_tensor = torch.ones(100, 100, device=device_hpu) * 7
    output_tensor_list = [torch.zeros(100, 100, device=device_hpu) for _ in range(world_size)]
    dist.all_gather(output_tensor_list, input_tensor, async_op=True).wait()

    for tensor in output_tensor_list:
        torch.testing.assert_close(tensor, input_tensor)

    # test all_reduce
    input_tensor = torch.ones(100, 100, device=device_hpu) * 7
    dist.all_reduce(input_tensor, async_op=True).wait()
    torch.testing.assert_close(input_tensor, torch.ones(100, 100, device=device_hpu) * (7 * world_size))

    # test broadcast
    if rank ==0:
        input_tensor = torch.ones(100, 100, device=device_hpu)
    else:
        input_tensor = torch.zeros(100, 100, device=device_hpu)
    dist.broadcast(input_tensor, 0, async_op=True).wait()
    torch.testing.assert_close(torch.ones(100, 100, device=device_hpu), input_tensor)

    # test reduce_scatter
    output_tensor = torch.zeros(100, 100, device=device_hpu)
    input_tensor_list = [torch.ones(100, 100, device=device_hpu) for _ in range(world_size)]
    dist.reduce_scatter(output_tensor, input_tensor_list, async_op=True).wait()
    torch.testing.assert_close(output_tensor, torch.zeros(100, 100, device=device_hpu) + world_size)

    dist.barrier()
    cleanup()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test_eager_collective_asycn test for veriying async op')
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    args = parser.parse_args()
    if args.verbose:
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
        os.environ["TORCH_SHOW_CPP_STACKTRACES"]="1"
    WORLD_SIZE = habana_frameworks.torch.hpu.device_count()
    if (WORLD_SIZE > 1):
        mp.spawn(simple,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True)
