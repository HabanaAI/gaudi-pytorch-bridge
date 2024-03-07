# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import habana_frameworks.torch.hpu as ht
import pytest
import torch
from test_utils import cpu, format_tc, hpu

slice_backward_test_case_list = [
    # size, dim, start, end, step
    ((4, 2), 0, 1, 4, 2),
    ((20, 10, 30, 4), 1, 2, 9, 3),
    ((3, 4, 5), 2, 0, 4, 1),
    ((4, 2), 0, 1, 2, 1),
    ((4, 2), 0, -3, -1, 1),
    ((4, 2), 0, 4, 4, 1),
    ((1, 1, 1), 2, 0, 8, 1),
    ((1, 1, 1), 1, 0, -1, 1),
    ((1, 35, 3), 2, 0, 8, 1),
    ((1, 35, 3), 1, 0, -1, 1),
]


@pytest.mark.parametrize("size, dim, start, end, step", slice_backward_test_case_list, ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32], ids=format_tc)
def test_slice_backward(size, dim, start, end, step, dtype):
    if dtype == torch.int32:
        fwd_input_cpu = torch.randint(-5000, 5000, dtype=dtype, size=size)
    else:
        fwd_input_cpu = torch.rand(size, dtype=dtype)

    fwd_output_cpu = torch.ops.aten.slice(fwd_input_cpu, dim, start, end, step)

    bwd_output_cpu = torch.ops.aten.slice_backward(fwd_output_cpu, fwd_input_cpu.size(), dim, start, end, step)

    fwd_output_hpu = fwd_output_cpu.to(hpu)

    bwd_output_hpu = torch.ops.aten.slice_backward(fwd_output_hpu, fwd_input_cpu.size(), dim, start, end, step)

    assert torch.equal(bwd_output_cpu, bwd_output_hpu.to(cpu))


@pytest.mark.parametrize("size", slice_backward_test_case_list[0], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=format_tc)
def test_slice_backward_autograd(size, dtype):
    fwd_input_cpu = torch.rand(size, dtype=dtype, requires_grad=True)

    def fn(tensor, device):
        tensor = tensor.to(device)
        tensor[..., tensor.shape[-1] :].sum().backward()
        return tensor

    bwd_output_cpu = fn(fwd_input_cpu, cpu)
    bwd_output_hpu = fn(fwd_input_cpu, hpu)

    assert torch.equal(bwd_output_cpu, bwd_output_hpu.to(cpu))
