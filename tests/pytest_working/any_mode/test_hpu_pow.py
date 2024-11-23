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
    is_gaudi1,
    is_lazy,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
integer_dtypes = []
if not is_gaudi1():
    dtypes += [torch.half]
if not is_lazy():
    integer_dtypes += [torch.int, torch.int16, torch.uint8, torch.int8]


def generate_inputs(shape, dtype):
    if dtype in integer_dtypes:
        low = 0 if dtype == torch.uint8 else -2
        high = 2
        cpu_input = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")

    return cpu_input, hpu_input


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_pow_tensor(shape, dtype):
    def fn(base, other):
        return torch.pow(base, other)

    cpu_input, hpu_input = generate_inputs(shape, dtype)
    cpu_other = torch.ones(size=shape, dtype=dtype) * 2
    hpu_other = cpu_other.to("hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = torch.pow(cpu_input, cpu_other)
    hpu_output = fn(hpu_input, hpu_other)

    atol = 1e-1 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-05)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("pow")
        clear_t_compile_logs()
        torch._dynamo.reset()

    cpu_output_scalar = torch.pow(cpu_input, 2)
    hpu_output_scalar = fn(hpu_input, 2)

    compare_tensors(hpu_output_scalar, cpu_output_scalar, atol=atol, rtol=2e-05)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("pow")


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_square(shape, dtype):
    def fn(base):
        return torch.square(base)

    cpu_input, hpu_input = generate_inputs(shape, dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = torch.square(cpu_input)
    hpu_output = fn(hpu_input)

    atol = 1e-1 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-05)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("pow")
