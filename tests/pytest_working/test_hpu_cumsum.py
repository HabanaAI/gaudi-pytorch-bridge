###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import habana_frameworks.torch
import torch
import torch.nn as nn
from test_utils import cpu, hpu


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, dim):
        t = torch.cumsum(x, dim)
        return t


from habana_frameworks.torch.hpu import random as hpu_random

seed = 42
torch.manual_seed(seed)
hpu_random.manual_seed_all(seed)


def test_cumsum_op():
    input_shapes = [
        (10, (4, 4), 0),
        (20, (5, 2), 1),
        (5, (2, 1), 0),
        (100, (200, 201), 0),
    ]

    model_hpu = Model().to(hpu)
    model_cpu = Model().to(cpu)
    for shape in input_shapes:
        # Test cumsum op with long datatype
        # If INT64 is not enabled in HPU, Long maps to i32 in TPC (in eager mode)
        x = torch.randint(shape[0], shape[1], dtype=torch.long, requires_grad=False, device=cpu)
        logits_hpu = model_hpu(x.to(hpu), shape[2])
        logits_cpu = model_cpu(x, shape[2])
        assert torch.allclose(logits_hpu.to(cpu), logits_cpu, atol=0.001, rtol=0.001)
