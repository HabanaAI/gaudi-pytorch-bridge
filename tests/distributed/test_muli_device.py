import multiprocessing
import os

import habana_frameworks.torch.core as htcore
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device("hpu")
torch.manual_seed(0)
input = [torch.randn(2000, 2000), torch.randn(2000, 2000), torch.randn(2000, 2000), torch.randn(2000, 2000)]
label = [torch.randn(2000, 2000), torch.randn(2000, 2000), torch.randn(2000, 2000), torch.randn(2000, 2000)]


def example(rank, world_size, port, devices):
    torch.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["HABANA_VISIBLE_MODULES"] = devices
    # create default process group
    print("Process created ", rank, world_size, port, devices)

    dist.init_process_group("hccl", rank=rank, world_size=world_size)

    # create local model
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(device) for _ in range(2)])
    model = model.to(device)
    # construct DDP model
    import copy

    ddp_model = DDP(copy.deepcopy(model).to(device), bucket_cap_mb=50, gradient_as_bucket_view=True)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01, weight_decay=1e-2, eps=1e-8)
    i = 1
    while i != 100:
        # forward pass
        outputs = ddp_model(input[rank].to(device))
        labels = label[rank].to(device)
        # backward pass
        loss = loss_fn(outputs, labels)
        loss.backward()
        # break the graph
        htcore.mark_step()
        # update parameters
        optimizer.step()
        htcore.mark_step()
        if rank == 0:
            print(" loss for step ", i, " is ", loss.to("cpu"))
        i = i + 1

    print(f"params sum is: {sum(model.parameters()).sum()}")


def main():
    world_size = 4
    p1 = []
    p2 = []
    for i in range(0, 4):
        p1.append(multiprocessing.Process(target=example, args=(i, world_size, "12345", "0,1,2,3")))
        p1[i].start()

    for i in range(0, 4):
        p2.append(multiprocessing.Process(target=example, args=(i, world_size, "12347", "4,5,6,7")))
        p2[i].start()

    for i in range(0, 4):
        p1[i].join()
        p2[i].join()


if __name__ == "__main__":
    main()
