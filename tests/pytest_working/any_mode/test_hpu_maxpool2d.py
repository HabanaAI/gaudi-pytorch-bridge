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

import os

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from habana_frameworks.torch.utils.debug.dynamo_utils import FxGraphAnalyzer
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    format_tc,
    is_pytest_mode_compile,
    setup_teardown_env_fixture,
)


@pytest.mark.parametrize(
    "shapes", [[(2, 9, 8), (2, 3, 16), (2, 3, 56)], [(1, 2, 9, 8), (1, 3, 3, 16), (1, 7, 3, 56)]], ids=format_tc
)
@pytest.mark.parametrize("kernel_size", [(4, 3)], ids=format_tc)
@pytest.mark.parametrize("stride", [(2, 1)], ids=format_tc)
@pytest.mark.parametrize("padding", [(2, 0)], ids=format_tc)
@pytest.mark.parametrize("dilation", [(1, 1)], ids=format_tc)
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
    def fn(model, input):
        results = model(input)
        results_wrap = results[0] if return_indices else results
        grad = torch.ones_like(results_wrap)
        results_wrap.backward(grad)

    maxpool2d = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    hpu_wrapped_fn = fn

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend")

    for shape in shapes:
        cpu_input = torch.rand(shape, dtype=dtype)
        hpu_input = cpu_input.to("hpu")
        cpu_input.requires_grad = True
        hpu_input.requires_grad = True

        fn(maxpool2d, cpu_input)
        hpu_wrapped_fn(maxpool2d, hpu_input)
        assert torch.allclose(cpu_input.grad, hpu_input.grad.to("cpu"))

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"max_pool2d_with_indices", "max_pool2d_with_indices_backward"})
