import time

import habana_frameworks.torch as ht
import torch
from torch import nn


def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        # torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        # if m.bias is not None:
        #   torch.nn.init.constant_(m.bias.data, 0)
        torch.nn.init.constant_(m.weight.data, 0.2)
        torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight.data, 1)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        #   torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.constant_(m.weight.data, 0.2)
        torch.nn.init.constant_(m.bias.data, 0.1)


def print_grads(model):
    for name, param in model.named_parameters():
        print(name, param.data)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(20, 20)
        self.output = nn.Linear(20, 20)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu_(x)
        for _ in range(10):
            x = self.sigmoid(x)
            x = self.output(x)
        x = self.softmax(x)
        return x


def test_graph_training1():
    N, D_in, D_out = 20, 20, 20
    module1 = Network().to("hpu")

    loss_fn = torch.nn.MSELoss()

    # optimizer = FusedSGD(module1.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(module1.parameters(), lr=0.1)

    x = torch.full((N, D_in), 0.25, device="hpu")

    module1.apply(initialize_weights)
    # print_grads(module1)
    iterations = 150
    real_inputs = [torch.full_like(x, 0.45) for _ in range(iterations)]
    real_targets = [torch.full((N, D_out), 0.36, device="hpu") for _ in range(iterations)]
    ht.core.mark_step()
    loss = 0
    start = time.time()

    for data, target in zip(real_inputs, real_targets):
        optimizer.zero_grad(set_to_none=True)
        ht.core.mark_step()
        tmp = module1(data)
        loss = loss_fn(tmp, target)
        loss.backward()
        optimizer.step()
        ht.core.mark_step()
    ht.hpu.synchronize()
    print("Time taken: ", time.time() - start, "Loss:", loss)


if __name__ == "__main__":
    test_graph_training1()
