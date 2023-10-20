#!/usr/bin/python3

import habana_frameworks.torch
import torch


def test_fun():
    class MyMod1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.par = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input):
            return input+1, self.par


    class MyMod2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mymod1 = MyMod1()

        def forward(self, input):
            out_ = self.mymod1(input)
            return out_[0]*2


    inp = torch.tensor(1.0, requires_grad=True).to("hpu")
    model = MyMod2().to("hpu")

    for _ in range(3):
        model(inp).sum().backward()

    assert len(model.mymod1.par._backward_hooks) == 1 , f"actual len(model.mymod1.par._backward_hooks) = {len(model.mymod1.par._backward_hooks)}, expected = 1"
