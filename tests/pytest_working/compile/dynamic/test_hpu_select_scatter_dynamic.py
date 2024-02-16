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
torch._dynamo.config.specialize_int=False

# shape_src should be of the same size as torch.select(input_shape, dim,index)
# This equates to the second dimension of input_shape for dim=0
@pytest.mark.parametrize("shape_src", [(2)])
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("index", [0])
@pytest.mark.skip(reason="SW-176493")
def test_select_scatter(shape_src, dim, index):
    input_shapes = [
        (2, 2),
        (4, 2),
        (3, 2),
        (5, 2)
        (6, 2)
    ]
    # Created a mini graph for testing
    # add op -> select_scatter op -> mul op
    def wrapper_fn(t, t_src, dim, indices):
        t1 = t.add(t)
        t2 = t1.select_scatter(t_src, dim, index)
        t3 = t2.mul(5)
        return t3

    f_cpu = torch.compile(wrapper_fn)
    f_hpu = torch.compile(wrapper_fn, backend="aot_hpu_training_backend", dynamic=None)

    for shape in input_shapes:
        input_tensor = torch.rand(shape, requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape_src, requires_grad=False, device=cpu)

        y_cpu = f_cpu(
            input_tensor, src_tensor, dim, index
        )
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), dim, index)

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.001, rtol=0.001)
