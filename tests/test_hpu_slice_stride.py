import torch
import torch_hpu

torch_hpu.is_available()

def func1(t, dev) :
  a = t.to(dev)
  a[0] = 1
  b = a[::4]
  b.add_(1)
  b = b.to("cpu")
  return b

ca = torch.rand(5, device="cpu")
h = func1(ca, "hpu")
c = func1(ca, "cpu")

assert(torch.allclose(h, c, 0.001, 0.001))
