# ******************************************************************************
# Copyright (C) 2020-22 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import torch
import torch
import os
import torch.nn as nn
from test_utils import compare_tensors
from copy import deepcopy

from habana_frameworks.torch.utils.library_loader import load_habana_module
import habana_frameworks.torch.core as htcore
N = 8#2
C = 320#4
H = 256#3
W = 256#4
G = 32#2
a = torch.randn((N,C,H,W))
ah = a.detach().to('hpu')
ah.requires_grad = True
a.requires_grad = True
b = a + 1.0
bh = ah + 1.0
gnc = nn.GroupNorm(G,C)
gnh = deepcopy(gnc).to('hpu')#nn.GroupNorm(2,4).to('hpu')
rgnh = gnh(bh)
sumh = torch.sum(rgnh)
sumh.backward()
htcore.mark_step()
