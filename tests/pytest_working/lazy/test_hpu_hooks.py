#!/usr/bin/python3

import habana_frameworks.torch
import torch
from habana_frameworks.torch.core.torch_overwrites import _name_stack


def test_fun():
    class MyMod1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.par = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input):
            return input + 1, self.par

    class MyMod2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mymod1 = MyMod1()

        def forward(self, input):
            out_ = self.mymod1(input)
            return out_[0] * 2

    inp = torch.tensor(1.0, requires_grad=True).to("hpu")
    model = MyMod2().to("hpu")

    for _ in range(3):
        model(inp).sum().backward()

    assert (
        len(model.mymod1.par._backward_hooks) == 1
    ), f"actual len(model.mymod1.par._backward_hooks) = {len(model.mymod1.par._backward_hooks)}, expected = 1"


def test_module_hooks():
    class SingleLayerModel(torch.nn.Module):
        def __init__(self):
            super(SingleLayerModel, self).__init__()

        def forward(self, x):
            return self.fc(x)

    def pre_forward_hook_raise_exception(module, input):
        print("Inside pre-forward hook")
        raise RuntimeError("Exception raised in pre-forward hook")

    input = torch.randn(1, 10)
    # Create the model
    model = SingleLayerModel()

    model_hpu = model.to("hpu")
    model_hpu.eval()
    input_hpu = input.to("hpu")
    model_hpu.add_module("fc", torch.nn.Linear(10, 5))

    # Register the pre-forward hook
    model_hpu.fc.register_forward_pre_hook(pre_forward_hook_raise_exception)
    for i in range(20):
        try:
            with torch.no_grad():
                output_hpu = model_hpu(input_hpu)
        except:
            pass
    print(_name_stack)
    assert not _name_stack, "Test failed because deque is not empty"
