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
# torch.compile test code for select_scatter op
# Set environment variable PT_HPU_LAZY_MODE to 0

import pytest
import torch
from test_utils import cpu, hpu


@pytest.mark.parametrize(
    "shape, shape_src, dim, index",
    [
        pytest.param((2, 2), (2), 0, 0),
    ],
)
def test_select_scatter(shape, shape_src, dim, index):
    # Created a mini graph for testing
    # add op -> select_scatter op -> mul op
    def wrapper_fn(t, t_src, dim, indices):
        t1 = t.add(t)
        t2 = t1.select_scatter(t_src, dim, index)
        t3 = t2.mul(5)
        return t3

    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend")

    input_tensor = torch.rand(shape, requires_grad=False, device=cpu)
    src_tensor = torch.rand(shape_src, requires_grad=False, device=cpu)

    y_cpu = wrapper_fn(input_tensor, src_tensor, dim, index)
    y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), dim, index)

    assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.001, rtol=0.001)
