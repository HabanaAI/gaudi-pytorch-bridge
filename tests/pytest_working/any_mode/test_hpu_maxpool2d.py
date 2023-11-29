###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
import os
import habana_frameworks.torch.dynamo.compile_backend
from test_utils import setup_teardown_env_fixture, format_tc


@pytest.mark.parametrize(
    "shapes",
    [[(2, 9, 8), (2, 3, 16), (2, 3, 56)], [(1, 2, 9, 8), (1, 3, 3, 16), (1, 7, 3, 56)]], ids=format_tc
)
@pytest.mark.parametrize("kernel_size", [(4, 3)])
@pytest.mark.parametrize("stride", [(2, 1)])
@pytest.mark.parametrize("padding", [(2, 0)])
@pytest.mark.parametrize("dilation", [(1, 1)])
@pytest.mark.parametrize("return_indices", [True, False], ids=format_tc)
@pytest.mark.parametrize("ceil_mode", [True, False], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_maxpool2d_bwd(
    shapes,
    kernel_size,
    stride,
    padding,
    dilation,
    return_indices,
    ceil_mode,
    dtype,
    setup_teardown_env_fixture,
):
    maxpool2d = torch.nn.MaxPool2d(
        kernel_size, stride, padding, dilation, return_indices, ceil_mode
    )
    hpu_wrapped_fn = torch.compile(
        maxpool2d, backend="aot_hpu_training_backend"
    ) if pytest.mode == "compile" else maxpool2d

    cpu_wrapped_fn = torch.compile(
        maxpool2d
    ) if pytest.mode == "compile" else maxpool2d

    torch._dynamo.reset()
    for shape in shapes:
        cpu_input = torch.rand(shape, dtype=dtype)
        hpu_input = cpu_input.to("hpu")
        cpu_input.requires_grad = True
        hpu_input.requires_grad = True

        res_hpu = hpu_wrapped_fn(hpu_input)
        res_cpu = cpu_wrapped_fn(cpu_input)

        if return_indices == True:
            res_hpu = res_hpu[0]
            res_cpu = res_cpu[0]

        grad_hpu = torch.ones_like(res_hpu)
        res_hpu.backward(gradient=grad_hpu)

        grad_cpu = torch.ones_like(res_cpu)
        res_cpu.backward(gradient=grad_cpu)
        assert torch.allclose(cpu_input.grad, hpu_input.grad.to("cpu"))
