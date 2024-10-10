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
