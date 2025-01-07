###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1


@pytest.mark.parametrize("shape", [[8, 32, 16], [2, 8, 32, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 3), 1)], ids=format_tc)
@pytest.mark.parametrize("stride", [(1, 2), 1, []], ids=format_tc)
@pytest.mark.parametrize("dilation", [(1, 2), 1], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_max_pool2d(shape, kernel_size_and_padding, stride, dilation, dtype):
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
    clear_t_compile_logs()
    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)
    check_ops_executed_in_jit_ir("max_pool2d_with_indices")


@pytest.mark.parametrize("shape", [[8, 32, 16], [1, 8, 32, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2), 1)], ids=format_tc)
@pytest.mark.parametrize("stride", [(1, 2), 1, []], ids=format_tc)
@pytest.mark.parametrize("dilation", [(1, 2), 1], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_max_pool2d_bwd(shape, kernel_size_and_padding, stride, dilation, dtype):
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
    clear_t_compile_logs()
    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)
    check_ops_executed_in_jit_ir({"max_pool2d_with_indices", "max_pool2d_with_indices_backward"})


@pytest.mark.parametrize("shape", [[7, 8, 16, 16], [1, 7, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2, 2), (1, 1, 1))])
@pytest.mark.parametrize("stride", [[1, 2, 2]])
@pytest.mark.parametrize("dilation", [[1, 2, 2]])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_max_pool3d(shape, kernel_size_and_padding, stride, dilation, dtype):
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
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("shape", [[7, 8, 16, 16], [1, 7, 8, 16, 16]])
@pytest.mark.parametrize("kernel_size_and_padding", [((2, 2, 2), (1, 1, 1))])
@pytest.mark.parametrize("stride", [[1, 2, 2]])
@pytest.mark.parametrize("dilation", [[1, 2, 2]])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_max_pool3d_bwd(shape, kernel_size_and_padding, stride, dilation, dtype):
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
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)
