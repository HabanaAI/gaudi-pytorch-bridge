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
import torch
import pytest
import random
from test_utils import (
    clear_t_compile_logs,
    check_ops_executed_in_jit_ir,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int]
fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
if not is_gaudi1():
    dtypes += fp8_dtypes


def generate_tensor(shape, dtype):
    if dtype in fp8_dtypes:
        return torch.rand(shape).to(dtype)
    return (torch.randn(shape) * 3.0).to(dtype)


@pytest.mark.parametrize("func", [torch.add, torch.sub, torch.rsub])
@pytest.mark.parametrize(
    "shape_a, shape_b", [[(), ()], [(1,), (2,)], [(4, 4), (1, 1)], [(16, 12), (16, 12)]]
)
@pytest.mark.parametrize("alpha", [1, 3])
@pytest.mark.parametrize("dtype", dtypes)
def test_binary(func, shape_a, shape_b, alpha, dtype):
    def fn(input, other, alpha):
        return func(input, other, alpha=alpha)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

    input = generate_tensor(shape_a, dtype)
    other = generate_tensor(shape_b, dtype)

    input_hpu = input.to("hpu")
    other_hpu = other.to("hpu")
    if dtype in fp8_dtypes:
        input = input.float()
        other = other.float()

    expected = func(input, other, alpha=alpha)
    result = fn(input_hpu, other_hpu, alpha)

    if dtype in fp8_dtypes:
        result = result.to(dtype)

    if dtype == torch.bfloat16:
        tol = 0.05
    elif dtype in fp8_dtypes:
        tol = 0.25
    else:
        tol = 1e-06
    compare_tensors(result, expected, atol=tol, rtol=tol)
    if is_pytest_mode_compile():
        name = "add" if func == torch.add else "sub"
        check_ops_executed_in_jit_ir(name)
