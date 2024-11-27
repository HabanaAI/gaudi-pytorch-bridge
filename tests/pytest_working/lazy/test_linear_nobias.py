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

import pytest
import torch
from test_utils import compare_tensors


@pytest.mark.skip
def test_linear_nobias():
    out_features = 16
    in_features = 8
    s1 = 5
    bsz = 4
    seqlen = 3
    """
    m = nn.Linear(in_features, out_features).to('hpu')
    input = torch.randn(seqlen, bsz, s1, in_features).to('hpu')
    output = m(input)
    print('..............................output size = ',output.size())
    """
    residual = torch.randn((seqlen, bsz, s1, out_features), requires_grad=True).to(torch.float)
    rh = residual.detach().to("hpu")
    a = torch.randn((seqlen, bsz, s1, in_features), requires_grad=True).to(torch.float)
    ah = a.detach().to("hpu")
    ah.requires_grad = True
    b = torch.randn((out_features, in_features), requires_grad=True).to(torch.float)
    bh = b.detach().to("hpu")
    bh.requires_grad = True
    c = torch.randn((out_features), requires_grad=True).to(torch.float)
    ch = c.detach().to("hpu")
    ch.requires_grad = True
    dh = torch.nn.functional.linear(ah, bh)
    print(dh.shape)
    rh + dh
    d = torch.nn.functional.linear(a, b)
    compare_tensors(dh.to("cpu"), d, rtol=1e-3, atol=1e-3)
    lcpu = d.sum()
    lcpu.backward()
    lhpu = dh.sum()
    lhpu.backward()
    compare_tensors(ah.grad.to("cpu"), a.grad, rtol=1e-3, atol=1e-3)
    compare_tensors(bh.grad.to("cpu"), b.grad, rtol=1e-3, atol=1e-3)
