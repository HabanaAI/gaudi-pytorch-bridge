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
# torch.compile Dynamic Shapes test code for select_scatter op
# Set environment variable PT_HPU_LAZY_MODE to 0

import pytest
import torch
from test_utils import cpu, hpu

torch._dynamo.config.specialize_int = False


# shape_src should be of the same size as torch.select(input_shape, dim,index)
# This equates to the second dimension of input_shape for dim=0
def test_select_scatter():
    # (input_shape, shape_src, dim, index)
    input_shapes = [
        ((16, 16), (16), 0, 4),
        ((16, 16), (16), 1, 8),
        ((16, 16), (16), 0, 5),
        ((16, 16), (16), 1, 6),
    ]

    # Created a mini graph for testing
    # add op -> select_scatter op -> mul op
    def wrapper_fn(t, t_src, dim, indices):
        t1 = t.add(t)
        t2 = t1.select_scatter(t_src, dim, indices)
        t3 = t2.mul(5)
        return t3

    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend", dynamic=None)

    for shape in input_shapes:
        input_tensor = torch.rand(shape[0], requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape[1], requires_grad=False, device=cpu)

        y_cpu = wrapper_fn(input_tensor, src_tensor, shape[2], shape[3])
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), shape[2], shape[3])

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.001, rtol=0.001)
