import torch
import habana_frameworks.torch.core as htcore
device = torch.device("hpu")

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

def test_shallow_copy_free():
    def fn(x, dev):
        y = x.add(1.0)
        x.data = torch.empty(0, dtype = x.dtype).to(dev)
        z = y.add(1.0)
        return y

    #CPU
    a = torch.randn([2, 3])
    ha = a.to('hpu')

    res = fn(a, 'cpu')
    print("cpu ", res)
    hres =  fn(ha, 'hpu')
    print("hpu ", hres.cpu())

    assert(torch.allclose(res, hres.cpu()))