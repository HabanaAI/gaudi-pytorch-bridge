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

#slice of slice
def func2(dev):
  torch.manual_seed(0)
  t = torch.rand(12,13,14).to(dev)
  k = t[1:11:2]
  k[:,2:5].add_(1)
  return k

cpu = func2("cpu").to("cpu")
hpu = func2("hpu").to("cpu")
assert(torch.allclose(hpu,cpu,0.001,0.001))

