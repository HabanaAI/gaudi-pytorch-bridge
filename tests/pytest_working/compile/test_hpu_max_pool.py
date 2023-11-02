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
from test_utils import is_torch_at_least
from pytest_working.test_utils import is_gaudi1

@pytest.mark.parametrize("shape", [[8, 16, 16], [1, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2), 1)])
@pytest.mark.parametrize("stride", [(1, 2), 1, []])
@pytest.mark.parametrize("dilation", [(1, 2), 1])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_max_pool2d(
    shape, kernel_size_and_padding, stride, dilation, dtype
):
    def fn(input):
        return torch.ops.aten.max_pool2d(
            input,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[8, 16, 16], [1, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2), 1)])
@pytest.mark.parametrize("stride", [(1, 2), 1, []])
@pytest.mark.parametrize("dilation", [(1, 2), 1])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_max_pool2d_bwd(
    shape, kernel_size_and_padding, stride, dilation, dtype
):
    if len(shape) == 3:
        pytest.xfail("[SW-164128] Missing 3D support")

    def fn(input):
        max_pool_2d = torch.ops.aten.max_pool2d(
            input,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )
        grad = torch.ones_like(max_pool_2d)
        max_pool_2d.backward(grad)
        return input.grad

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[7, 8, 16, 16], [1, 7, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2, 2), (1, 1, 1))])
@pytest.mark.parametrize("stride", [[1, 2, 2]])
@pytest.mark.parametrize("dilation", [[1, 2, 2]])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_max_pool3d(
    shape, kernel_size_and_padding, stride, dilation, dtype
):
    def fn(input):
        return torch.ops.aten.max_pool3d(
            input,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[7, 8, 16, 16], [1, 7, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2, 2), (1, 1, 1))])
@pytest.mark.parametrize("stride", [[1, 2, 2]])
@pytest.mark.parametrize("dilation", [[1, 2, 2]])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_max_pool3d_bwd(
    shape, kernel_size_and_padding, stride, dilation, dtype
):
    if len(shape) == 4:
        pytest.xfail("[SW-164128] Missing 4D support")
    if is_gaudi1() == True:
        pytest.xfail("[SW-165533] result mismatch")
    def fn(input):
        max_pool_3d = torch.ops.aten.max_pool3d(
            input,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )
        grad = torch.ones_like(max_pool_3d)
        max_pool_3d.backward(grad)
        return input.grad

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)
