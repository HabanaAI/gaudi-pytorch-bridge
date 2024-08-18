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
# torch.compile Dynamic Shapes test code for slice_scatter op
# Set environment variable PT_HPU_LAZY_MODE to 0

import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import cpu, hpu

torch._dynamo.config.specialize_int = False


# Test case to check slice_scatter DS with negative values as parameters
# SW-180452
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_slice_scatter_negative_end():
    input_shapes = [
        ((2, 17, 16, 3706), (2, 17, 16, 3702), 3, 0, -4),
        ((2, 17, 20, 3706), (2, 17, 16, 3706), 2, 0, -4),
        ((2, 17, 20, 3706), (2, 17, 20, 3706), 1, 0, 9223372036854775807),
        ((3, 17, 20, 3706), (2, 17, 20, 3706), 0, 0, -1),
    ]

    # Created a mini graph for testing
    # add op -> slice_scatter op -> mul op
    def wrapper_fn(t, t_src, dim, start, end, step):
        t1 = t.add(t)
        t2 = t1.slice_scatter(t_src, dim, start, end, step)
        t3 = t2.mul(5)
        return t3

    f_cpu = torch.compile(wrapper_fn, dynamic=True)
    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend", dynamic=None)

    for shape in input_shapes:
        input_tensor = torch.rand(shape[0], requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape[1], requires_grad=False, device=cpu)

        y_cpu = f_cpu(input_tensor, src_tensor, shape[2], shape[3], shape[4], 1)
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), shape[2], shape[3], shape[4], 1)

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.01, rtol=0.01)


# shape_src should be of the same size as torch.select(input_shape, dim,index)
# This equates to the second dimension of input_shape for dim=0
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_slice_scatter_compatible_with_select_scatter():
    input_shapes = [
        ((16, 16), (1, 16), 0, 4, 5, 1),
        ((16, 16), (1, 16), 0, 3, 4, 1),
        ((16, 16), (16, 1), 1, 9, 10, 1),
        ((16, 12), (1, 12), 0, 7, 8, 1),
    ]

    # Created a mini graph for testing
    # add op -> slice_scatter op -> mul op
    def wrapper_fn(t, t_src, dim, start, end, step):
        t1 = t.add(t)
        t2 = t1.slice_scatter(t_src, dim, start, end, step)
        t3 = t2.mul(5)
        return t3

    f_cpu = torch.compile(wrapper_fn)
    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend", dynamic=None)

    for shape in input_shapes:
        input_tensor = torch.rand(shape[0], requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape[1], requires_grad=False, device=cpu)

        y_cpu = f_cpu(input_tensor, src_tensor, shape[2], shape[3], shape[4], shape[5])
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), shape[2], shape[3], shape[4], shape[5])

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.01, rtol=0.01)


# Test skipped due to issues reported at SW-177340
@pytest.mark.skip(reason="SW-177340")
def test_slice_scatter():
    input_shapes = [
        ((16, 16), (16, 1), 1, 4, 6, 2),
        ((16, 16), (16, 1), 1, 3, 4, 1),
        ((16, 16), (16, 1), 1, 10, 15, 1),
        ((16, 16), (1, 16), 0, 6, 14, 1),
    ]

    # Created a mini graph for testing
    # add op -> slice_scatter op -> mul op
    def wrapper_fn(t, t_src, dim, start, end, step):
        t1 = t.add(t)
        t2 = t1.slice_scatter(t_src, dim, start, end, step)
        t3 = t2.mul(5)
        return t3

    f_cpu = torch.compile(wrapper_fn)
    f_hpu = torch.compile(wrapper_fn, backend="hpu_backend", dynamic=None)

    for shape in input_shapes:
        input_tensor = torch.rand(shape[0], requires_grad=False, device=cpu)
        src_tensor = torch.rand(shape[1], requires_grad=False, device=cpu)

        y_cpu = f_cpu(input_tensor, src_tensor, shape[2], shape[3], shape[4], shape[5])
        y_hpu = f_hpu(input_tensor.to(hpu), src_tensor.to(hpu), shape[2], shape[3], shape[4], shape[5])

        assert torch.allclose(y_cpu, y_hpu.to(cpu), atol=0.01, rtol=0.01)
