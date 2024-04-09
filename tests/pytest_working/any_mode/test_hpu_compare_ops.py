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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1, is_pytest_mode_compile

compare_ops_out = ["lt", "gt", "ge"]
compare_ops_inplace = ["lt_", "gt_", "ge_"]
compare_ops = [*compare_ops_out, *compare_ops_inplace]
integer_types = [torch.int, torch.int8, torch.long]
supported_dtypes = [*integer_types, torch.float32, torch.bfloat16]
if not is_gaudi1():
    supported_dtypes.append(torch.float16)


def fn_out(op, input, other, out=None):
    return op(input, other, out=out)


def fn_inplace(op, input, other):
    return op(input, other)


def get_wrapped_fns(inplace: bool):
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        return (
            (fn_inplace, torch.compile(fn_inplace, backend="hpu_backend"))
            if inplace
            else (fn_out, torch.compile(fn_out, backend="hpu_backend"))
        )
    else:
        return (fn_inplace, fn_inplace) if inplace else (fn_out, fn_out)


@pytest.mark.parametrize("op_name", compare_ops)
@pytest.mark.parametrize("shape", [[1], [10, 10]], ids=format_tc)
@pytest.mark.parametrize("dtype", supported_dtypes, ids=format_tc)
def test_hpu_compare_op_tensor(op_name, shape, dtype):
    inplace = op_name in compare_ops_inplace
    op = getattr(torch.Tensor if inplace else torch, op_name)

    cpu_input = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    cpu_other = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    hpu_other = cpu_other.to("hpu")

    cpu_fn, hpu_wrapped_fn = get_wrapped_fns(inplace)

    cpu_output = cpu_fn(op, cpu_input, cpu_other)
    hpu_output = hpu_wrapped_fn(op, hpu_input, hpu_other).cpu()
    assert torch.equal(cpu_output, hpu_output)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name[:-1] if inplace else op_name)


@pytest.mark.parametrize("op_name", compare_ops)
@pytest.mark.parametrize("shape", [[1], [10, 10]], ids=format_tc)
@pytest.mark.parametrize("dtype", supported_dtypes, ids=format_tc)
def test_hpu_compare_op_scalar(op_name, shape, dtype):
    inplace = op_name in compare_ops_inplace
    op = getattr(torch.Tensor if inplace else torch, op_name)

    cpu_input = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    other = 0

    cpu_fn, hpu_wrapped_fn = get_wrapped_fns(inplace)

    cpu_output = cpu_fn(op, cpu_input, other)
    hpu_output = hpu_wrapped_fn(op, hpu_input, other).cpu()
    assert torch.equal(cpu_output, hpu_output)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name[:-1] if inplace else op_name)


@pytest.mark.parametrize("op_name", compare_ops_out)
@pytest.mark.parametrize("shape", [[1], [10, 10]], ids=format_tc)
@pytest.mark.parametrize("dtype", supported_dtypes, ids=format_tc)
def test_hpu_compare_op_tensor_out(op_name, shape, dtype):
    op = getattr(torch, op_name)

    cpu_input = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    cpu_other = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    cpu_output = torch.empty(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    hpu_other = cpu_other.to("hpu")
    hpu_output = torch.empty(shape, dtype=dtype, device="hpu")

    cpu_fn, hpu_wrapped_fn = get_wrapped_fns(False)

    cpu_fn(op, cpu_input, cpu_other, cpu_output)
    hpu_wrapped_fn(op, hpu_input, hpu_other, hpu_output)
    hpu_output = hpu_output.cpu()
    assert torch.equal(cpu_output, hpu_output)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name)


@pytest.mark.parametrize("op_name", compare_ops_out)
@pytest.mark.parametrize("shape", [[1], [10, 10]], ids=format_tc)
@pytest.mark.parametrize("dtype", supported_dtypes, ids=format_tc)
def test_hpu_compare_op_scalar_out(op_name, shape, dtype):
    op = getattr(torch, op_name)

    cpu_input = (
        torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        if dtype in integer_types
        else torch.randn(shape, dtype=dtype)
    )
    cpu_output = torch.empty(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    hpu_output = torch.empty(shape, dtype=dtype, device="hpu")
    other = 0

    cpu_fn, hpu_wrapped_fn = get_wrapped_fns(False)

    cpu_fn(op, cpu_input, other, cpu_output)
    hpu_wrapped_fn(op, hpu_input, other, hpu_output)
    hpu_output = hpu_output.cpu()
    assert torch.equal(cpu_output, hpu_output)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name)
