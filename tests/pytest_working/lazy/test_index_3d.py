###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import torch
from test_utils import compare_tensors, cpu, hpu

s0 = 4
s1 = 3
s2 = 3


def index_original(device) -> torch.Tensor:
    x = torch.arange(s0 * s1 * s2, device=device).view(s0, s1, s2)
    # x = torch.arange(s0*s1*s2).view(s0, s1, s2).to(device)
    torch.Tensor([0, 2]).to(device).to(torch.int64)
    # z = torch.Tensor([1, 2]).to(device).to(torch.int64)
    torch.Tensor([1, 2]).to(device).to(torch.int64)
    torch.Tensor([1]).to(device).to(torch.int64)
    bmask1 = torch.tensor([False, True, False]).to(device)
    return x[:, torch.tensor([0, 1]).to("cpu"), bmask1]


def test_index_3d():
    index_hpu = (index_original(hpu).to("cpu"),)
    index_cpu = index_original(cpu)
    compare_tensors(index_hpu, index_cpu, atol=0, rtol=0)
