# python test_async_collectives.py

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from habana_frameworks.torch.utils.library_loader import load_habana_module

torch.manual_seed(0)
load_habana_module()
device = torch.device("hpu")


def setup(rank, world_size):
    print("Setup")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12340"
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)


def cleanup():
    print("Cleanup")
    dist.destroy_process_group()


def async_allReduce():
    _tensor = torch.ones(10).to(device)
    torch.distributed.all_reduce(_tensor)
    _tensor_cpu = _tensor.cpu()


def main_worker(gpu, world_size):
    setup(gpu, world_size)

    for i in range(100):
        print("Iteration", i)
        async_allReduce()

    cleanup()


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.linear1 = nn.Linear(30, 20)
        self.eval()

    def forward(self, x):
        x = self.conv(x)  # Assuming the input is 1D, unsqueeze to make it 2D
        x = self.relu(x)
        torch.distributed.all_reduce(x)
        x = self.linear1(x.view(x.size(0), -1))  # Flatten the output before linear layer
        return x


def main_allreduce_worker(gpu, world_size):
    setup(gpu, world_size)
    # Dummy input for testing
    input = torch.randn(1, 32, 1, 10)

    model = CustomModel()
    model.eval()

    input_hpu = input.to(device)
    model_hpu = model.to(device)
    model_hpu.eval()

    with torch.no_grad():
        output_hpu = model_hpu(input_hpu)
        output_hpu_cpu = output_hpu.cpu()

    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    hpugraph_module = wrap_in_hpu_graph(model_hpu, disable_tensor_cache=True)
    import habana_frameworks.torch.core as htcore

    htcore.hpu_set_inference_env()
    with torch.no_grad():
        output_hpugraph = hpugraph_module(input_hpu)
        output_hpugraph_cpu = output_hpugraph.cpu()

    assert torch.allclose(output_hpu_cpu, output_hpugraph_cpu, atol=0.001, rtol=0.001)


# Use command line
# PT_HPU_ENABLE_LAZY_COLLECTIVES=1 python tests/distributed/test_async_collectives.py
if __name__ == "__main__":
    n_gpus = 8
    mp.spawn(main_allreduce_worker, args=(n_gpus,), nprocs=n_gpus, join=True)
    mp.spawn(main_worker, args=(n_gpus,), nprocs=n_gpus, join=True)
