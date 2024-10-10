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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    hpu,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.bfloat16, torch.float]
if not is_gaudi1():
    dtypes.append(torch.float16)


#  _add_relu on CPU doesn't support bfloat16 nor float16
def add_relu_cpu(input, other, alpha):
    if input.dtype == torch.float:
        return torch._add_relu(input, other, alpha=alpha)
    return torch.relu(input + other * alpha)


def _test_add_relu(input, other, alpha, is_scalar):
    def add_relu(input, other, alpha):
        return torch._add_relu(input, other, alpha=alpha)

    # Force scalars to be BF16/F16
    if is_scalar:
        other = torch.tensor(other, dtype=input.dtype).item()
    alpha = torch.tensor(alpha, dtype=input.dtype).item()

    result_cpu = add_relu_cpu(input, other, alpha)
    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        add_relu = torch.compile(add_relu, backend="hpu_backend", dynamic=False)

    other = other if is_scalar else other.to(hpu)
    result_hpu = add_relu(input.to(hpu), other, alpha)
    compare_tensors(result_hpu, result_cpu, rtol=1e-5, atol=1e-5)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_add_relu")


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("alpha", [1.0, -0.26, 0.723], ids=format_tc)
@pytest.mark.parametrize("other", [-2.0, 4.3], ids=format_tc)
@pytest.mark.parametrize("input_shape", [(5, 3, 1, 2), (1), (2, 3, 7)], ids=format_tc)
def test_add_relu_scalar(input_shape, other, alpha, dtype):
    input = torch.rand(input_shape, dtype=dtype)
    _test_add_relu(input, other, alpha, is_scalar=True)


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("alpha", [1.0, -0.26, 0.723], ids=format_tc)
@pytest.mark.parametrize(
    "in_out_shapes",
    [
        [(5, 3, 1, 2), (1, 3, 5, 2)],
        [(5, 4, 1, 2), (1, 2)],
        [(2, 1, 7), (1, 3, 7)],
    ],
    ids=format_tc,
)
def test_add_relu_tensor(in_out_shapes, alpha, dtype):
    input_shape, other_shape = in_out_shapes
    input = torch.rand(input_shape, dtype=dtype)
    other = torch.rand(other_shape, dtype=dtype)
    _test_add_relu(input, other, alpha, is_scalar=False)
