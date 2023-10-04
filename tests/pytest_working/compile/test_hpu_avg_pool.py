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
import habana_frameworks.torch.dynamo.compile_backend

@pytest.mark.parametrize("shape", [[2, 7], [2, 2, 7]])
@pytest.mark.parametrize("kernel_size_and_padding", [(1, 0), (2, 0), (2,1), (3, 1)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_avg_pool1d(shape, kernel_size_and_padding, stride, dtype):
    def fn(input):
        return torch.ops.aten.avg_pool1d(input, kernel_size, stride=stride, padding=padding)

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.equal(cpu_output, hpu_output)

@pytest.mark.parametrize("shape", [[2, 7], [2, 2, 7]])
@pytest.mark.parametrize("output_size", [1, 6, 10])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_adaptive_avg_pool1d(shape, output_size, dtype):
    def fn(input):
        return torch.ops.aten.adaptive_avg_pool1d(input, output_size)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).to("cpu")
    assert torch.allclose(cpu_output, hpu_output)

@pytest.mark.parametrize("shape", [[1, 8, 16, 16], [1, 1, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((3, 2, 2), 1), (4, (1, 2, 2)), ((1, 1, 1), 0)])
@pytest.mark.parametrize("stride", [(2, 1, 2), 1, 2])
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
@pytest.mark.parametrize("divisor_override", [None, 4, -3])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_avg_pool3d(shape, kernel_size_and_padding, stride, ceil_mode, count_include_pad, divisor_override, dtype):
    if divisor_override != None and divisor_override < 0:
        pytest.xfail('[SW-160805] Negative divisors error')
    if (ceil_mode==True
        and count_include_pad==True
        and kernel_size_and_padding==((3, 2, 2), 1)
        and divisor_override==None
        and (stride==(2,1,2) or stride==2)):
        pytest.xfail('[SW-160805] Output mismatch on last dim')

    def fn(input):
        return torch.ops.aten.avg_pool3d(
            input,
            kernel_size,
            padding=padding,
            stride=stride,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override)

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)

@pytest.mark.parametrize("shape", [[8, 16, 16], [1, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2), 1), ((4, 4), 2)])
@pytest.mark.parametrize("stride", [(1, 2), 1, 2])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_avg_pool2d_bwd(shape, kernel_size_and_padding, stride, dtype):
    if shape == [8, 16, 16]:
        pytest.xfail('[SW-161411] bwd kernel does not support 3d input')
    def fn(input):
        input.requires_grad = True
        avg_pool = torch.ops.aten.avg_pool2d(input, kernel_size=kernel_size, padding=padding, stride=stride)
        grad = torch.ones_like(avg_pool)
        avg_pool.backward(grad)
        return input.grad

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)