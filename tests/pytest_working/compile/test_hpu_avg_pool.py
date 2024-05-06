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
from test_utils import format_tc


@pytest.mark.parametrize("shape", [[2, 7], [2, 2, 7]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [(1, 0), (2, 0), (2, 1), (3, 1)], ids=format_tc)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_avg_pool1d(shape, kernel_size_and_padding, stride, dtype):
    def fn(input):
        return torch.ops.aten.avg_pool1d(input, kernel_size, stride=stride, padding=padding)

    torch._dynamo.reset()
    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", [[2, 7], [2, 2, 7]], ids=format_tc)
@pytest.mark.parametrize("output_size", [1, 6, 10])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_adaptive_avg_pool1d(shape, output_size, dtype):
    def fn(input):
        return torch.ops.aten.adaptive_avg_pool1d(input, output_size)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    torch._dynamo.reset()
    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).to("cpu")
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[1, 8, 16, 16], [1, 1, 8, 16, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [((3, 2, 2), 1), (4, (1, 2, 2)), ((1, 1, 1), 0)], ids=format_tc)
@pytest.mark.parametrize("stride", [(2, 1, 2), 1, 2], ids=format_tc)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
@pytest.mark.parametrize("divisor_override", [None, 4, -3])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_avg_pool3d(shape, kernel_size_and_padding, stride, ceil_mode, count_include_pad, divisor_override, dtype):
    def fn(input):
        return torch.ops.aten.avg_pool3d(
            input,
            kernel_size,
            padding=padding,
            stride=stride,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

    torch._dynamo.reset()
    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[8, 16, 16], [1, 8, 16, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2), 1)], ids=format_tc)
@pytest.mark.parametrize("stride", [(1, 2), 1], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_avg_pool2d(shape, kernel_size_and_padding, stride, dtype):
    def fn(input):
        return torch.ops.aten.avg_pool2d(input, kernel_size=kernel_size, padding=padding, stride=stride)

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[8, 16, 16], [1, 8, 16, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2), 1), ((4, 4), 2)], ids=format_tc)
@pytest.mark.parametrize("stride", [(1, 2), 1, 2], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_avg_pool2d_bwd(shape, kernel_size_and_padding, stride, dtype):
    def fn(input):
        avg_pool = torch.ops.aten.avg_pool2d(input, kernel_size=kernel_size, padding=padding, stride=stride)
        grad = torch.ones_like(avg_pool)
        avg_pool.backward(grad)
        return input.grad

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[1, 2, 3, 7], [1, 1, 2, 3, 7]], ids=format_tc)
@pytest.mark.parametrize("output_size", [(2, 3, 1), (2, 3, 6), (2, 3, 10)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_adaptive_avg_pool3d(shape, output_size, dtype):
    def fn(input):
        return torch.ops.aten.adaptive_avg_pool3d(input, output_size)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).to("cpu")
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[1, 2, 3, 7], [1, 1, 2, 3, 7], [4, 8, 7, 7], [2, 4, 8, 7, 7]], ids=format_tc)
@pytest.mark.parametrize("output_size", [(2, 3, 1), (2, 3, 6), (2, 3, 10)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_adaptive_avg_pool3d_bwd(shape, output_size, dtype):
    def fn(input):
        fwd = torch.ops.aten.adaptive_avg_pool3d(input, output_size)
        grad = torch.ones_like(fwd)
        fwd.backward(grad)
        return input.grad

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input)
    assert torch.allclose(cpu_output, hpu_output.cpu())


@pytest.mark.parametrize("shape", [[8, 16, 16], [1, 8, 16, 16]], ids=format_tc)
@pytest.mark.parametrize("output_size", [((2, 2))], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_adaptive_avg_pool2d_bwd(shape, output_size, dtype):
    def fn(input):
        avg_pool = torch.ops.aten.adaptive_avg_pool2d(input, output_size)
        grad = torch.ones_like(avg_pool)
        avg_pool.backward(grad)
        return input.grad

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[2, 2, 7], [2, 2, 2, 7]], ids=format_tc)
@pytest.mark.parametrize("output_size", [(2, 10)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_adaptive_avg_pool2d(shape, output_size, dtype):
    def fn(input):
        return torch.ops.aten.adaptive_avg_pool2d(input, output_size)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).to("cpu")
    assert torch.allclose(cpu_output, hpu_output)
