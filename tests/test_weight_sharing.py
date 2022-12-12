import torch
import numpy as np
import inspect
import os
try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    assert False, "Could Not import habana_frameworks.torch.core"


def test_hpu_weight_sharing_in_same_module():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.zeros([1]))
            self.b = torch.nn.Parameter(torch.ones([1]))
            self.c = torch.nn.Parameter(torch.ones([1]))
        def forward(self, input):
            c = self.a/input + self.b*input * self.c
            return c

    #initial - cpu, no weight sharing
    model = TestModel()
    result = model(1)
    assert model.a.device.type == "cpu"
    assert model.b.device.type == "cpu"
    assert model.c.device.type == "cpu"
    assert np.equal(result.detach(), 1)

    #weight sharing
    model.a = model.b
    result = model(2)
    assert np.equal(result.cpu().detach(), 2.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.cpu().detach(), 1)
    assert np.equal(model.c.cpu().detach(), 1)

    #casting to hpu
    model.to("hpu")
    result = model(2)
    loss = result.sum()
    loss.backward()
    assert model.a.device.type == "hpu"
    assert model.b.device.type == "hpu"
    assert model.c.device.type == "hpu"
    assert np.equal(model.a.grad.cpu().detach(), model.b.grad.cpu().detach())
    assert np.equal(result.cpu().detach(), 2.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.cpu().detach(), 1)
    assert np.equal(model.c.cpu().detach(), 1)

def test_hpu_weight_sharing_in_submodule():
    class TestSubModel(torch.nn.Module):
        def __init__(self, shared_parameter):
            super(TestSubModel, self).__init__()
            self.a = torch.nn.Parameter(torch.zeros([1]))
            self.a = shared_parameter
        def forward(self, input):
            c = self.a/input
            return c

    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.ones([1]))
            self.b = TestSubModel(self.a)
        def forward(self, input):
            c = self.a/input + self.b(input)*input
            return c
    # initial - cpu with weight sharing
    model = TestModel()
    result = model(2)
    assert model.a.device.type == "cpu"
    assert model.b.a.device.type == "cpu"
    assert np.equal(result.detach(), 1.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.a.cpu().detach(), 1)

    #casting to hpu
    model.to("hpu")
    result = model(2)
    loss = result.sum()
    loss.backward()
    assert model.a.device.type == "hpu"
    assert model.b.a.device.type == "hpu"
    assert np.equal(model.a.grad.cpu().detach(), model.b.a.grad.cpu().detach())
    assert np.equal(result.cpu().detach(), 1.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.a.cpu().detach(), 1)

def test_hpu_weight_sharing_in_exported_parameter():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.ones([1]))

    #initial - cpu with exported parameter
    model = TestModel()
    exported = model.a
    assert model.a.device.type == "cpu"
    assert exported.device.type == "cpu"
    assert np.equal(model.a.cpu().detach() == exported.cpu().detach(), model.a.cpu().detach() == 1)
    exported.data += 1
    assert np.equal(model.a.cpu().detach() == exported.cpu().detach(), model.a.cpu().detach() == 2)
    model.a.data += 1
    assert np.equal(model.a.cpu().detach() == exported.cpu().detach(), model.a.cpu().detach() == 3)

    #hpu with exported parameter
    model.to("hpu")
    assert model.a.device.type == "hpu"
    assert exported.device.type == "hpu"
    assert np.equal(model.a.cpu().detach() == exported.cpu().detach(), model.a.cpu().detach() == 3)
    exported.data += 1
    assert np.equal(model.a.cpu().detach() == exported.cpu().detach(), model.a.cpu().detach() == 4)
    model.a.data += 1
    assert np.equal(model.a.cpu().detach() == exported.cpu().detach(), model.a.cpu().detach() == 5)

def test_hpu_workaround_for_cpu_caching_without_weight_sharing():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.ones([1]))

    model = TestModel()
    model.to("hpu")
    model.a.data = model.a.data.cpu()

def test_hpu_weight_sharing_in_exported_parameter_cpu_hpu_cpu():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.ones([1]))

    model = TestModel()
    nadav_list = model.a

    assert model.a.device.type == "cpu"
    assert nadav_list.device.type == "cpu"

    model.to("hpu")
    assert model.a.device.type == "hpu"
    assert nadav_list.device.type == "hpu"

    model.to("cpu")
    assert model.a.device.type == "cpu"
    assert nadav_list.device.type == "cpu"

    model.to("hpu")
    assert model.a.device.type == "hpu"
    assert nadav_list.device.type == "hpu"

def test_hpu_weight_sharing_in_same_module_cpu_hpu_cpu():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.zeros([1]))
            self.b = torch.nn.Parameter(torch.ones([1]))
        def forward(self, input):
            c = self.a/input + self.b*input
            return c

    #initial - cpu, no weight sharing
    model = TestModel()
    result = model(1)
    assert model.a.device.type == "cpu"
    assert model.b.device.type == "cpu"
    assert np.equal(result.detach(), 1)

    #weight sharing
    model.a = model.b
    result = model(2)
    assert np.equal(result.cpu().detach(), 2.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.cpu().detach(), 1)

    model.to("hpu")
    assert model.a.device.type == "hpu"
    assert model.b.device.type == "hpu"

    model.to("cpu")
    result = model(2)
    loss = result.sum()
    loss.backward()
    assert model.a.device.type == "cpu"
    assert model.b.device.type == "cpu"
    assert np.equal(model.a.grad.cpu().detach(), model.b.grad.cpu().detach())
    assert np.equal(result.cpu().detach(), 2.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.cpu().detach(), 1)

def test_hpu_multiple_weight_sharing_in_same_module():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.zeros([1]))
            self.b = torch.nn.Parameter(torch.ones([1]))
            self.c = torch.nn.Parameter(torch.ones([1]))
            self.d = torch.nn.Parameter(torch.ones([1]))
        def forward(self, input):
            c = self.a/input + self.b*input + self.b/input + self.d*input
            return c

    #initial - cpu, no weight sharing
    model = TestModel()
    result = model(1)
    assert model.a.device.type == "cpu"
    assert model.b.device.type == "cpu"
    assert model.c.device.type == "cpu"
    assert model.d.device.type == "cpu"
    assert np.equal(result.detach(), 3)

    #weight sharing
    model.a = model.b
    model.c = model.b
    model.d = model.b
    result = model(2)
    assert np.equal(result.cpu().detach(), 5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.cpu().detach(), 1)
    assert np.equal(model.c.cpu().detach(), 1)
    assert np.equal(model.d.cpu().detach(), 1)

    #casting to hpu
    model.to("hpu")
    result = model(2)
    loss = result.sum()
    loss.backward()
    assert model.a.device.type == "hpu"
    assert model.b.device.type == "hpu"
    assert np.equal(model.a.grad.cpu().detach(), model.b.grad.cpu().detach())
    assert np.equal(result.cpu().detach(), 5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.cpu().detach(), 1)
    assert np.equal(model.c.cpu().detach(), 1)
    assert np.equal(model.d.cpu().detach(), 1)

def test_hpu_multiple_weight_sharing_in_submodule():
    class TestSubModel(torch.nn.Module):
        def __init__(self, shared_parameter):
            super(TestSubModel, self).__init__()
            self.a = torch.nn.Parameter(torch.zeros([1]))
            self.b = torch.nn.Parameter(torch.zeros([1]))
            self.a = shared_parameter
            self.b = shared_parameter
        def forward(self, input):
            c = self.a/input + self.b/input
            return c

    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.a = torch.nn.Parameter(torch.ones([1]))
            self.b = TestSubModel(self.a)
        def forward(self, input):
            c = self.a/input + self.b(input)*input
            return c
    # initial - cpu with weight sharing
    model = TestModel()
    result = model(2)
    assert model.a.device.type == "cpu"
    assert model.b.a.device.type == "cpu"
    assert np.equal(result.detach(), 2.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.a.cpu().detach(), 1)
    assert np.equal(model.b.b.cpu().detach(), 1)

    #casting to hpu
    model.to("hpu")
    result = model(2)
    loss = result.sum()
    loss.backward()
    assert model.a.device.type == "hpu"
    assert model.b.a.device.type == "hpu"
    assert np.equal(model.a.grad.cpu().detach(), model.b.a.grad.cpu().detach())
    assert np.equal(model.a.grad.cpu().detach(), model.b.b.grad.cpu().detach())
    assert np.equal(result.cpu().detach(), 2.5)
    assert np.equal(model.a.cpu().detach(), 1)
    assert np.equal(model.b.a.cpu().detach(), 1)
    assert np.equal(model.b.b.cpu().detach(), 1)