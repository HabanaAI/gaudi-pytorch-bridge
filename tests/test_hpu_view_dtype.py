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

import pytest
import torch
from test_utils import cpu, hpu


# Tests input tensors
def func1(dev, sdtype, ddtype):

    a = torch.rand(1, 2, dtype=sdtype).to(dev)
    b = a.view(ddtype)
    return b


# Tests intermediate tensors
def func2(dev, sdtype, ddtype):

    a = torch.rand(3, 2, dtype=sdtype).to(dev)
    b = torch.rand(3, 2, dtype=sdtype).to(dev)
    c = a * b
    d = c.view(ddtype)
    return d


# Tests inplace ops
def func3(dev, sdtype, ddtype):

    a = torch.rand(3, 2, dtype=sdtype).to(dev)
    b = torch.rand(3, 2, dtype=sdtype).to(dev)
    c = a * b
    d = c.view(ddtype)
    c.add_(1)
    return d


@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("func", [func1, func2, func3])
@pytest.mark.parametrize("dtype1, dtype2", [(torch.float32, torch.int32), (torch.bfloat16, torch.int32)])
def test_hpu_view(func, dtype1, dtype2):
    cpu_result = func(cpu, dtype1, dtype2)
    hpu_result = func(hpu, dtype1, dtype2)
    assert torch.allclose(cpu_result, hpu_result.to(cpu))
