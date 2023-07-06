import torch
import torch_hpu

torch_hpu.is_available()


def test_hpu_slice():
    def func1(t, dev):
        a = t.to(dev)
        a[0] = 1
        b = a[::4]
        b.add_(1)
        b = b.to("cpu")
        return b

    ca = torch.rand(5, device="cpu")
    h = func1(ca, "hpu")
    c = func1(ca, "cpu")

    assert torch.allclose(h, c, 0.001, 0.001)

    # slice of slice
    def func2(dev):
        t = torch.rand(12, 13, 14).to(dev)
        k = t[1:11:2]
        k[:, 2:5].add_(1)
        return k

    cpu = func2("cpu").to("cpu")
    hpu = func2("hpu").to("cpu")
    assert torch.allclose(hpu, cpu, 0.001, 0.001)

    def func3(dev, k):

        t = torch.zeros(12, 13, 14).to(dev)
        t[1:11:k].add_(1)
        t[:, 1:11:k].add_(1)
        return t

    cpu1 = func3("cpu", 2).to("cpu")
    cpu2 = func3("cpu", 3).to("cpu")
    hpu1 = func3("hpu", 2).to("cpu")
    hpu2 = func3("hpu", 3).to("cpu")

    assert torch.allclose(hpu1, cpu1, 0.001, 0.001)
    assert torch.allclose(hpu2, cpu2, 0.001, 0.001)
