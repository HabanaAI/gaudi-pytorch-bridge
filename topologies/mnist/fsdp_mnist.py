import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

import habana_frameworks.torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import functools

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import habana_frameworks.torch as ht
device_hpu = torch.device('hpu')


def setup(rank, world_size):
   os.environ['MASTER_ADDR'] = 'localhost'
   os.environ['MASTER_PORT'] = '12355'

   # initialize the process group
   import habana_frameworks.torch.distributed.hccl
   dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def cleanup():
   dist.destroy_process_group()

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, 3, 1)
       self.conv2 = nn.Conv2d(32, 64, 3, 1)
       self.dropout1 = nn.Dropout(0.25)
       self.dropout2 = nn.Dropout(0.5)
       self.fc1 = nn.Linear(9216, 128)
       self.fc2 = nn.Linear(128, 10)

   def forward(self, x):
       x = self.conv1(x)
       x = F.relu(x)
       x = self.conv2(x)
       x = F.relu(x)
       x = F.max_pool2d(x, 2)
       x = self.dropout1(x)
       x = torch.flatten(x, 1)
       x = self.fc1(x)
       x = F.relu(x)
       x = self.dropout2(x)
       x = self.fc2(x)
       output = F.log_softmax(x, dim=1)
       return output

g_wrap_param_size = 200

def train(args, model, device, rank, world_size, train_loader, optimizer, epoch, sampler=None, prof=None):
   model.train()

   ddp_loss = torch.zeros(2).to(device)
   if sampler:
       sampler.set_epoch(epoch)
   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.to(device), target.to(device)
       optimizer.zero_grad()
       output = model(data)
       loss = F.nll_loss(output, target, reduction='sum')
       loss.backward()
       optimizer.step()
       ddp_loss[0] += loss.item()
       ddp_loss[1] += len(data)
       if prof:
           prof.step()

   dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
   if rank == 0:
       print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, device, rank, world_size, test_loader):
   model.eval()
   correct = 0
   ddp_loss = torch.zeros(3).to(device)
   with torch.no_grad():
       for data, target in test_loader:
           data, target = data.to(device), target.to(device)
           output = model(data)
           ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
           pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
           ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
           ddp_loss[2] += len(data)
           dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)

   if rank == 0:
       test_loss = ddp_loss[0] / ddp_loss[2]
       print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
           test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
           100. * ddp_loss[1] / ddp_loss[2]))


def fsdp_main(rank, world_size, args):
   setup(rank, world_size)
   device = torch.device("hpu")

   transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
       ])
   global g_wrap_param_size

   dataset1 = datasets.MNIST('./data', train=True, download=True,
                      transform=transform)
   dataset2 = datasets.MNIST('./data', train=False,
                      transform=transform)

   sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
   sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

   train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
   test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
   hpu_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': False}
   train_kwargs.update(hpu_kwargs)
   test_kwargs.update(hpu_kwargs)

   train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
   test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
   my_auto_wrap_policy = functools.partial(
           size_based_auto_wrap_policy, min_num_params=g_wrap_param_size)
   # make it clear if cpu_offload running or not
   print(f"--> CPUOffload is set to {args.use_offload}, in rank {rank}")

   init_start_event = habana_frameworks.torch.hpu.Event(enable_timing=True)
   init_end_event = habana_frameworks.torch.hpu.Event(enable_timing=True)

   init_start_event.record()

   model = Net().to(device)

   #model =  DDP(model)
   model = FSDP(model,
                    auto_wrap_policy = size_based_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=args.use_offload)  # test cpu offload
            )

   #TODO fix issue
   #model = FSDP(model,
   #             auto_wrap_policy = my_auto_wrap_policy,
   #             cpu_offload=CPUOffload(offload_params=args.use_offload))

   optimizer = optim.Adam(model.parameters(), lr=args.lr)

   scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
   if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA,
                                                torch.profiler.ProfilerActivity.HPU],
                                    schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler('fsdp_mnist/profile_traces'),
                                    profile_memory=True,
                                    with_stack=False,
                                    record_shapes=True) as prof:

            if rank==0:
                print("\n--> Profiling active...")

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1, prof=prof)
                test(model, device, rank, world_size, test_loader)
                scheduler.step()

   else:
      if rank==0:
          print("\n--> * Not * profiling\n")

      for epoch in range(1, args.epochs + 1):
          train(args, model, device, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
          test(model, device, rank, world_size, test_loader)
          scheduler.step()

   init_end_event.record()
   time = init_start_event.elapsed_time(init_end_event) / 1000
   if rank == 0:
       print("Elapsed time in sec", time)

   if args.save_model and rank == 0:
       torch.save(model.state_dict(), "mnist_cnn.pt")

   if args.save_model:
       print("Warning - saving is only available on nightlies atm")
       # use a barrier to make sure training is done on all ranks
       dist.barrier()
       # state_dict for FSDP model is only available on Nightlies for now
       states = model.state_dict()
       if rank == 0:
          torch.save(states, "fsdp_mnist.pt")

   cleanup()

if __name__ == '__main__':
   # Training settings
   parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
   parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                       help='input batch size for training (default: 64)')
   parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                       help='input batch size for testing (default: 1000)')
   parser.add_argument('--epochs', type=int, default=2, metavar='N',
                       help='number of epochs to train (default: 14)')

   parser.add_argument('--lr', type=float, default=.003, metavar='LR',
                       help='learning rate (default: 1.0)')

   parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                       help='Learning rate step gamma (default: 0.7)')
   parser.add_argument('--seed', type=int, default=1, metavar='S',
                       help='random seed (default: 1)')
   parser.add_argument('--save-model', action='store_true', default=False,
                       help='For Saving the current Model')
   parser.add_argument('--use_offload', action='store_true', default=False, help='whether CPUOffload is enabled')
   parser.add_argument('--profile', action='store_true', default=False, help='enable profiling')
   args = parser.parse_args()

   torch.manual_seed(args.seed)

   WORLD_SIZE = habana_frameworks.torch.hpu.device_count()
   os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
   mp.spawn(fsdp_main,
       args=(WORLD_SIZE, args),
       nprocs=WORLD_SIZE,
       join=True)
#cmnd to run
#  PT_HPU_EAGER_PIPELINE_ENABLE=false PT_HPU_LAZY_MODE=0 PT_HPU_COMPILE_USE_RECIPES=True python3 -u fsdp_mnist.py







