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
bool_dtype = [torch.bool] if not is_lazy() else []
if not is_gaudi1():
    dtypes += [torch.half]
if not is_lazy():
    integer_dtypes += [torch.int, torch.int16, torch.int8, torch.uint8]


def generate_inputs(shape, dtype):
    if dtype in integer_dtypes + bool_dtype:
        low = 0 if dtype in [torch.bool, torch.uint8] else -2
        high = 2
        cpu_input = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")

    return cpu_input, hpu_input


@pytest.mark.parametrize("op_name", ["prod", "nansum"])
@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes + bool_dtype, ids=format_tc)
def test_hpu_reduction(op_name, shape, dtype):
    op = getattr(torch, op_name)

    def fn(input, dtype):
        return op(input, dtype=dtype)

    cpu_input, hpu_input = generate_inputs(shape, dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = op(cpu_input, dtype=dtype)
    hpu_output = fn(hpu_input, dtype)

    atol = 1e-1 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-5)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name)


@pytest.mark.parametrize("op_name", ["prod", "nansum", "amin", "amax"])
@pytest.mark.parametrize(
    "shape_and_dim",
    [([2, 7], None), ([2, 7], 1), ([2, 3, 4], None), ([2, 3, 4], 2), ([2, 3, 4], (0, 1)), ([2, 3, 4], (2, 1, 0))],
)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes + bool_dtype, ids=format_tc)
def test_hpu_reduction_dim(op_name, shape_and_dim, keepdim, dtype):
    op = getattr(torch, op_name)
    shape, dim = shape_and_dim
    if op_name == "prod" and (type(dim) is tuple or dim is None):
        pytest.skip("torch.prod doesn't support tuple/None as dim parameter")

    def fn(input, dim, keepdim, **kargs):
        return op(input, dim=dim, keepdim=keepdim, **kargs)

    cpu_input, hpu_input = generate_inputs(shape, dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    if op_name in ["amax", "amin"]:
        cpu_output = op(cpu_input, dim=dim, keepdim=keepdim)
        hpu_output = fn(hpu_input, dim, keepdim)
        atol, rtol = 0, 0
    else:
        cpu_output = op(cpu_input, dim=dim, keepdim=keepdim, dtype=dtype)
        hpu_output = fn(hpu_input, dim, keepdim, dtype=dtype)
        atol = 1e-1 if dtype in [torch.bfloat16, torch.half] else 1e-4
        rtol = 2e-5

    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(op_name)


prod_dtypes = [torch.float32, torch.bfloat16, torch.int, torch.int16]
if not is_gaudi1():
    prod_dtypes += [torch.half, torch.long]


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", prod_dtypes)
def test_hpu_prod(shape, dtype):
    def fn(input):
        return torch.prod(input, dtype=dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_input, hpu_input = generate_inputs(shape, dtype)

    cpu_output = torch.prod(cpu_input, dtype=dtype)
    hpu_output = fn(hpu_input)

    atol = 1e-2 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-5)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("prod")


@pytest.mark.parametrize("shape", [(4, 4, 4)])
@pytest.mark.parametrize("dtype", prod_dtypes)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_hpu_prod_dim(shape, dtype, dim, keepdim):
    def fn(input):
        return torch.prod(input, dim, keepdim=keepdim, dtype=dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_input, hpu_input = generate_inputs(shape, dtype)

    cpu_output = torch.prod(cpu_input, dim, keepdim=keepdim, dtype=dtype)
    hpu_output = fn(hpu_input)

    atol = 1e-2 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-5)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("prod")


@pytest.mark.parametrize("shape", [(4, 4, 4)])
@pytest.mark.parametrize("dtype", prod_dtypes)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_hpu_prod_out(shape, dtype, dim, keepdim):

    def fn(input, out):
        torch.ops.aten.prod.int_out(input, dim, keepdim=keepdim, dtype=dtype, out=out)
        return

    cpu_input, hpu_input = generate_inputs(shape, dtype)
    cpu_output = cpu_input.new_empty((cpu_input.shape[0], cpu_input.shape[1]))
    hpu_output = hpu_input.new_empty((hpu_input.shape[0], hpu_input.shape[1]))
    if keepdim:
        cpu_output = cpu_output.unsqueeze(dim)
        hpu_output = hpu_output.unsqueeze(dim)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    # inplace op for cpu
    torch.ops.aten.prod.int_out(cpu_input, dim, keepdim=keepdim, dtype=dtype, out=cpu_output)

    # inplace op for hpu
    fn(hpu_input, hpu_output)

    atol = 1e-1 if dtype in [torch.bfloat16, torch.half] else 1e-4
    compare_tensors(hpu_output, cpu_output, atol=atol, rtol=2e-5)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("prod")
