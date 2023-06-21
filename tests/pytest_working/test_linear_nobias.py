###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
import torch
import os
import torch.nn as nn
from test_utils import compare_tensors

from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

out_features = 16
in_features = 8
s1 = 5
bsz = 4
seqlen = 3
'''
m = nn.Linear(in_features, out_features).to('hpu')
input = torch.randn(seqlen, bsz, s1, in_features).to('hpu')
output = m(input)
print('..............................output size = ',output.size())
'''
residual = torch.randn((seqlen, bsz, s1, out_features), requires_grad=True).to(torch.float)
rh = residual.detach().to('hpu')
a = torch.randn((seqlen, bsz, s1, in_features), requires_grad=True).to(torch.float)
ah = a.detach().to('hpu')
ah.requires_grad = True
b = torch.randn((out_features, in_features), requires_grad=True).to(torch.float)
bh = b.detach().to('hpu')
bh.requires_grad = True
c = torch.randn((out_features), requires_grad=True).to(torch.float)
ch = c.detach().to('hpu')
ch.requires_grad = True
dh = torch.nn.functional.linear(ah,bh)
#print(dh.to('cpu'))
print(dh.shape)
res = rh + dh
#print(res.to('cpu'))
#print(res.shape)
d = torch.nn.functional.linear(a,b)
comp_res = compare_tensors(dh.to("cpu"), d, rtol=1e-3, atol=1e-3)
print('cpu and hpu result match = ',comp_res)
lcpu = d.sum()
print(lcpu)
lcpu.backward()
lhpu = dh.sum()
print('...............lhpu = ',lhpu.to('cpu'), dh.requires_grad)
lhpu.backward()
print(ah.grad.to('cpu'))
bwdres1 = compare_tensors(ah.grad.to("cpu"), a.grad, rtol=1e-3, atol=1e-3)
print('t1: cpu and hpu result match = ',bwdres1)
bwdres2 = compare_tensors(bh.grad.to("cpu"), b.grad, rtol=1e-3, atol=1e-3)
print('t2: cpu and hpu result match = ',bwdres2)
print('b = ',b.grad)
print('bh = ',bh.grad.to('cpu'))
'''
bwdres3 = compare_tensors(ch.grad.to("cpu"), c.grad, rtol=1e-3, atol=1e-3)
print('t3: cpu and hpu result match = ',bwdres3)
print('c = ',c.grad)
print('ch = ',ch.grad.to('cpu'))
'''
