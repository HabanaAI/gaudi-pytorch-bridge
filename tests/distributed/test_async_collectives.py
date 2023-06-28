# python test_async_collectives.py

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from habana_frameworks.torch.utils.library_loader import load_habana_module

torch.manual_seed(0)
load_habana_module()
device = torch.device('hpu')

def setup(rank, world_size):
  print('Setup')
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12340'
  dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def cleanup():
  print('Cleanup')
  dist.destroy_process_group()

def async_allReduce():
  _tensor = torch.ones(10).to(device)
  torch.distributed.all_reduce(_tensor)
  _tensor_cpu = _tensor.cpu()

def main_worker(gpu, world_size):
  setup(gpu, world_size)

  for i in range(100):
    print('Iteration', i)
    async_allReduce()

  cleanup()

if __name__ == '__main__':
  n_gpus = 8
  mp.spawn(main_worker, args=(n_gpus,), nprocs=n_gpus, join=True)
