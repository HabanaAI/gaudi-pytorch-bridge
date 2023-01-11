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

def func1(dev, sdtype, ddtype):
   torch.manual_seed(0)
   a = torch.rand(1,2, dtype=sdtype).to(dev)
   b = a.view(ddtype)
   return b

cpu = func1("cpu", torch.float32, torch.int32)
hpu = func1("hpu", torch.float32, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))

cpu = func1("cpu", torch.bfloat16, torch.int32)
hpu = func1("hpu", torch.bfloat16, torch.int32)
assert(torch.allclose(cpu, hpu.to('cpu')))
