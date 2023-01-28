import torch
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()
device = torch.device("hpu")
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.p = nn.Parameter(torch.rand([50264, 64], dtype=torch.float32, device=device))

    def forward(self, x):
        x = F.relu(self.conv1)
        return F.relu(self.conv2)

myModel = Model().to('hpu')

offset = 0
offset_next = 3216896
bucket = torch.empty([3242176], dtype=torch.float32, device=device)
with torch.no_grad():
     bucket[offset:offset_next].copy_(myModel.p.data.flatten())
     myModel.p.data = bucket[offset:offset_next].view_as(myModel.p.data)
myModel.p.data += 1
print(torch.equal(myModel.p.data.flatten(), bucket[offset:offset_next]))

def test_simple():
    def func(dev):
        base = torch.tensor([1,2,3,4,5,6], device=dev)
        view = base[:]
        base.data = view.data
        return base, view

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert(torch.equal(cpu_tensor, hpu_tensor.cpu()))

def test_two_shallow_copies():
    def func(dev):
        base = torch.tensor([1,2,3,4,5,6], device=dev)
        base2 = torch.tensor([0], device=dev)
        base2.data = base.data
        view = base2[:]
        base2.data = view.data
        return base2, base, view

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert(torch.equal(cpu_tensor, hpu_tensor.cpu()))

def test_view_with_strides():
    def func(dev):
        base = torch.tensor([1,2,3,4,5,6], device=dev)
        view = base[::2]
        base.data = view.data
        return base, view

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert(torch.equal(cpu_tensor, hpu_tensor.cpu()))

def test_view_with_strides2():
    def func(dev):
        base = torch.tensor([1,2,3,4,5,6], device=dev).add(2.0)
        view = base[::2]
        view2 = base.view(-1)
        base.data = view.data

        return base.mul_(2.0), view.add(2.0), view2

    for cpu_tensor, hpu_tensor in zip(func("cpu"), func("hpu")):
        assert(torch.equal(cpu_tensor, hpu_tensor.cpu()))

test_simple()
test_two_shallow_copies()
test_view_with_strides()
test_view_with_strides2()