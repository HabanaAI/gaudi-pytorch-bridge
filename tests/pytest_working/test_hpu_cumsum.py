###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
