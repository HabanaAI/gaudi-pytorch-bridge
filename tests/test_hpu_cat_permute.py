import torch
from test_utils import *

a = torch.randn(2,3,4,4).to("hpu")
d = torch.randn(2,3,4,4).to("hpu")
wt = torch.randn(1,1,3,3).to("hpu")
b = torch.nn.functional.conv2d(a,wt)
e = torch.nn.functional.conv2d(d,wt)
c = torch.cat([b,e], dim=1)
c.to(cpu)
print(c.size())
