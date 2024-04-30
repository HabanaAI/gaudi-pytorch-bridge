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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("shape", [tuple(), (3, 3)])
@pytest.mark.parametrize("dim", [None, (-1, -2), 0])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("p", [None, "fro", "nuc", 0, 1, 2])
@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16])
def test_hpu_norm(shape, dim, keepdim, p, dtype):
    if len(shape) == 0 and dim != 0:
        pytest.skip("Unsupported test configuration")
    if p == "nuc" and (len(shape) == 0 or (not isinstance(dim, tuple) or len(dim) != 2)):
        pytest.skip("Unsupported test configuration")
    if p == "nuc" and shape == (3, 3) and dim == (-1, -2):
        pytest.skip("Unsupported test configuration (aten::_linalg_svd.U is not yet supported on HPU)")

    def fn(input):
        if p == "fro" or p == "nuc":
            return torch.norm(input, p=p, dim=dim, keepdim=keepdim)
        else:
            return torch.norm(input, p=p, dim=dim, keepdim=keepdim, dtype=dtype)

    input_dtype = dtype if dtype != None else torch.bfloat16
    if p == "nuc":
        input_dtype = torch.float

    cpu_input = torch.tensor(2, dtype=input_dtype) if len(shape) == 0 else torch.rand(shape, dtype=input_dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [(2,)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_zero_sized_group_norm(shape, dtype):
    def fn(input, num_groups, weight, bias):
        fwd = torch.nn.functional.group_norm(input, num_groups, weight=weight, bias=bias)
        grad = torch.randn_like(fwd)
        fwd.backward(grad)
        return (input.grad, weight.grad, bias.grad)

    input = torch.randn(0, shape[0], dtype=dtype, requires_grad=True)
    weight = torch.rand(shape, dtype=dtype, requires_grad=True)
    bias = torch.rand(shape, dtype=dtype, requires_grad=True)
    input_hpu = input.detach().to("hpu")
    weight_hpu = weight.detach().to("hpu")
    bias_hpu = bias.detach().to("hpu")
    input_hpu.requires_grad_(True)
    weight_hpu.requires_grad_(True)
    bias_hpu.requires_grad_(True)

    hpu_wrapped_fn = fn
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend")
    output_cpu = fn(input, 1, weight, bias)
    output_hpu = hpu_wrapped_fn(input_hpu, 1, weight_hpu, bias_hpu)

    compare_tensors(output_hpu, output_cpu, atol=0, rtol=0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("native_group_norm")


@pytest.mark.parametrize("shape", [(2, 2, 2)], ids=format_tc)
def test_hpu_zero_sized_batch_norm(shape):
    def fn(input, running_mean, running_var, weight, bias):
        fwd = torch.nn.functional.batch_norm(input, running_mean, running_var, weight=weight, bias=bias)
        grad = torch.randn_like(fwd)
        fwd.backward(grad)
        return (input.grad, bias.grad)

    C = shape[0]

    input = torch.randn(0, *shape, requires_grad=True)
    running_mean = torch.randn(C)
    running_var = torch.randn(C) + 1
    weight = torch.randn(C, requires_grad=True) + 1
    bias = torch.randn(C, requires_grad=True)
    input_hpu = input.detach().to("hpu")
    running_mean_hpu = running_mean.detach().to("hpu")
    running_var_hpu = running_var.detach().to("hpu")
    weight_hpu = weight.detach().to("hpu")
    bias_hpu = bias.detach().to("hpu")
    input_hpu.requires_grad_(True)
    weight_hpu.requires_grad_(True)
    bias_hpu.requires_grad_(True)

    hpu_wrapped_fn = fn
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend")
    output_cpu = fn(input, running_mean, running_var, weight, bias)
    output_hpu = hpu_wrapped_fn(input_hpu, running_mean_hpu, running_var_hpu, weight_hpu, bias_hpu)

    compare_tensors(output_hpu, output_cpu, atol=0, rtol=0)
    if is_pytest_mode_compile():
        # batch norm is decomposed
        check_ops_executed_in_jit_ir({"sum", "full", "select_scatter", "mul", "full"})
