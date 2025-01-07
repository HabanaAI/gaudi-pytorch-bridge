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
from test_utils import _is_simulator


@pytest.mark.skipif(_is_simulator(), reason="high memory usage may couse problems on sim")
def test_add():
    # validating using the property a*X + b*X = (a+b)*X
    G = 1024 * 1024 * 1024
    shape, m, n = 2 * G, 12, 10
    HPU = torch.device("hpu")
    inp1 = torch.ones(shape, dtype=torch.float32, device=HPU) * m
    inp2 = torch.ones(shape, dtype=torch.float32, device=HPU) * n
    result_ref = torch.ones(shape, dtype=torch.float32, device=HPU) * (m + n)
    assert torch.equal(torch.add(inp1, inp2), result_ref)


@pytest.mark.skipif(_is_simulator(), reason="high memory usage may couse problems on sim")
def test_transpose():
    G = 1024 * 1024 * 1024
    shape = [G, 2]
    dim0 = 0
    dim1 = 1
    inp_t = torch.randn(shape, dtype=torch.float32, device=torch.device("hpu"))
    result = torch.transpose(inp_t, dim0, dim1)
    # validate data
    assert torch.equal(torch.transpose(result, dim0, dim1), inp_t)


@pytest.mark.skipif(_is_simulator(), reason="high memory usage may couse problems on sim")
def test_detach():
    G = 1024 * 1024 * 1024
    shape = [2 * G]
    t1 = torch.randn(shape, dtype=torch.float32, device="hpu")
    t2 = t1.detach()
    assert torch.equal(t1, t2)
