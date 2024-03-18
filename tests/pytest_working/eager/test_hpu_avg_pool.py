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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("shape", [[1, 8, 16, 16], [1, 1, 8, 16, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [(4, (1, 2, 2)), ((1, 1, 1), 0)], ids=format_tc)
@pytest.mark.parametrize("stride", [(2, 1, 2)], ids=format_tc)
@pytest.mark.parametrize("ceil_mode", [False])
@pytest.mark.parametrize("count_include_pad", [False])
@pytest.mark.parametrize("divisor_override", [None, 4, -3])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_avg_pool3d_bwd_grad_input(
    shape,
    kernel_size_and_padding,
    stride,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dtype,
):
    def fn(input):
        fwd = torch.ops.aten.avg_pool3d(
            input,
            kernel_size,
            padding=padding,
            stride=stride,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        grad = torch.ones_like(fwd)
        grad_input = torch.zeros_like(input)
        output = torch.ops.aten.avg_pool3d_backward.grad_input(
            grad,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            grad_input=grad_input,
        )
        return output

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    cpu_output = fn(cpu_input)
    hpu_output = fn(hpu_input)
    assert torch.allclose(cpu_output, hpu_output.cpu())
