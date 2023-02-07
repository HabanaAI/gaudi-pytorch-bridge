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
import torch_hpu
torch_hpu.is_available()

# Tests input tensors
def func1(dev, sdtype, ddtype):
   torch.manual_seed(0)
   a = torch.rand(1,2, dtype=sdtype).to(dev)
   b = a.view(ddtype)
   return b

# Tests intermediate tensors
def func2(dev, sdtype, ddtype):
   torch.manual_seed(0)
   a = torch.rand(3,2, dtype=sdtype).to(dev)
   b = torch.rand(3,2, dtype=sdtype).to(dev)
   c = a * b
   d = c.view(ddtype)
   return d

# Tests inplace ops
def func3(dev, sdtype, ddtype):
   torch.manual_seed(0)
   a = torch.rand(3,2, dtype=sdtype).to(dev)
   b = torch.rand(3,2, dtype=sdtype).to(dev)
   c = a * b
   d = c.view(ddtype)
   c.add_(1)
   return d

cpu = func1("cpu", torch.float32, torch.int32)
hpu = func1("hpu", torch.float32, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))
cpu = func1("cpu", torch.bfloat16, torch.int32)
hpu = func1("hpu", torch.bfloat16, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))

cpu = func2("cpu", torch.float32, torch.int32)
hpu = func2("hpu", torch.float32, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))
cpu = func2("cpu", torch.bfloat16, torch.int32)
hpu = func2("hpu", torch.bfloat16, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))

cpu = func3("cpu", torch.float32, torch.int32)
hpu = func3("hpu", torch.float32, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))
cpu = func3("cpu", torch.bfloat16, torch.int32)
hpu = func3("hpu", torch.bfloat16, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))
