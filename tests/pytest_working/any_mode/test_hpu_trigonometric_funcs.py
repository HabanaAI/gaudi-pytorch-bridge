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
from test_utils import compare_tensors, format_tc, is_lazy

dtypes = [torch.float32, torch.bfloat16]
integer_dtypes = []
if not is_lazy():
    integer_dtypes += [torch.int, torch.int16, torch.int8, torch.uint8]

ops = [torch.sin, torch.cos, torch.tanh]
inversed_ops = [torch.asin, torch.acos, torch.atanh]
hyperbolic_ops = [torch.sinh, torch.cosh]
inversed_hyperbolic_ops = [torch.asinh, torch.acosh]


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("op", ops + inversed_ops + hyperbolic_ops + inversed_hyperbolic_ops)
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_trigonometric_funcs(shape, op, dtype):
    def fn(input):
        return op(input)

    if dtype in integer_dtypes:
        if op in ops + hyperbolic_ops + [torch.atanh, torch.asinh]:
            low = 0 if dtype == torch.uint8 else -100
            high = 100
        elif op in inversed_ops:
            low = 0 if dtype == torch.uint8 else -1
            high = 2
        elif op == torch.acosh:
            low = 1
            high = 100
        cpu_input = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    else:
        if op in ops + hyperbolic_ops + [torch.atanh, torch.asinh]:
            low = -100
            high = 100
        elif op in inversed_ops:
            low = -1
            high = 1
        elif op == torch.acosh:
            low = 1
            high = 100
        cpu_input = torch.empty(shape).to(dtype)
        cpu_input.uniform_(low, high)

    hpu_input = cpu_input.to("hpu")

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = op(cpu_input)
    hpu_output = fn(hpu_input)

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-5)
