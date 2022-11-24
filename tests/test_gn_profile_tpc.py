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
import os
import torch.nn as nn
from test_utils import compare_tensors
from copy import deepcopy
from habana_frameworks.torch.utils.library_loader import load_habana_module
import habana_frameworks.torch.core as htcore
os.environ['PT_HPU_LAZY_MODE'] = "1"
device='hpu'

C = 32
for i in range(30):
    gnc = nn.GroupNorm(4, C)
    gnh = deepcopy(gnc).to('hpu')
    a = torch.ones((32, C, 128, 256), dtype=torch.float)
    ah = a.detach().to('hpu')
    ah.requires_grad = True
    a.requires_grad = True
    rgnh = gnh(ah)
    sumh = torch.sum(rgnh)
    sumh.backward()
    htcore.mark_step()
