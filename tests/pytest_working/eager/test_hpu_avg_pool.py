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
from test_utils import compare_tensors, format_tc, hpu, is_gaudi1


@pytest.mark.parametrize("shape", [[1, 8, 16, 16], [1, 1, 8, 16, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size_and_padding", [(4, (1, 2, 2)), ((1, 1, 1), 0)], ids=format_tc)
@pytest.mark.parametrize("stride", [(2, 1, 2)], ids=format_tc)
@pytest.mark.parametrize("ceil_mode", [False])
@pytest.mark.parametrize("count_include_pad", [False])
@pytest.mark.parametrize("divisor_override", [None, 4, -3])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_avg_pool3d_bwd_grad_input(
    shape,
    kernel_size_and_padding,
    stride,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dtype,
):
    def fn(input):
        fwd = torch.ops.aten.avg_pool3d(
            input,
            kernel_size,
            padding=padding,
            stride=stride,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

        grad = torch.ones_like(fwd)
        grad_input = torch.zeros_like(input)
        output = torch.ops.aten.avg_pool3d_backward.grad_input(
            grad,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            grad_input=grad_input,
        )
        return output

    kernel_size, padding = kernel_size_and_padding
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to(hpu)

    cpu_output = fn(cpu_input)
    hpu_output = fn(hpu_input)
    assert torch.allclose(cpu_output, hpu_output.cpu())


dtypes = [torch.bfloat16, torch.float]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("input_shape", [[1, 2, 3, 7], [1, 1, 2, 3, 7], [4, 8, 7, 7], [2, 4, 8, 7, 7]], ids=format_tc)
@pytest.mark.parametrize("output_shape", [[2, 3, 1], [2, 3, 6], [2, 3, 10]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_adaptive_avg_pool3d_bwd(input_shape, output_shape, dtype):
    def fn(input_shape, output_shape, dtype, device):
        grad = torch.ones(input_shape[:-3] + output_shape, dtype=dtype, device=device)
        input = torch.rand(input_shape, dtype=dtype, device=device)
        grad_input = torch.zeros(input_shape, dtype=dtype, device=device)
        torch.ops.aten.adaptive_avg_pool3d_backward(grad, input, grad_input=grad_input)
        return grad_input

    result_cpu = fn(input_shape, output_shape, dtype, device="cpu")
    result_hpu = fn(input_shape, output_shape, dtype, device=hpu)
    tol = 1e-4 if dtype == torch.float16 else 1e-5
    compare_tensors(result_hpu, result_cpu, rtol=tol, atol=tol)
