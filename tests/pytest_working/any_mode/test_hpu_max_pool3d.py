###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

dtype = torch.float32
dilation = [1, 2, 3]
stride = [3, 2, 1]


@pytest.mark.parametrize("shape", [[8, 16, 16, 16], [1, 1, 16, 16, 16]], ids=format_tc)
@pytest.mark.parametrize("kernel_size, padding", [((2, 2, 2), 0), ((3, 4, 2), (1, 0, 1))], ids=format_tc)
@pytest.mark.parametrize("ceil_mode", [True, False], ids=format_tc)
@pytest.mark.parametrize("is_out", [True, False], ids=format_tc)
def test_hpu_max_pool3d_with_indices_fwd_bwd(shape, kernel_size, padding, ceil_mode, is_out):
    if is_out and is_pytest_mode_compile:
        pytest.skip("Out variant is not supported in torch.compile")

    def fn(input):
        result = torch.empty(0, dtype=input.dtype, device=input.device)
        indices = torch.empty(0, dtype=torch.long, device=input.device)
        grad_input = torch.empty_like(input)

        if is_out:
            result, indices = torch.ops.aten.max_pool3d_with_indices(
                input,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                out=result,
                ceil_mode=ceil_mode,
                indices=indices,
            )

            grad_output = torch.ones_like(result)
            torch.ops.aten.max_pool3d_with_indices_backward(
                grad_output,
                input,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                indices,
                grad_input=grad_input,
            )
        else:
            result, indices = torch.ops.aten.max_pool3d_with_indices(
                input,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )

            grad_output = torch.ones_like(result)
            grad_input = torch.ops.aten.max_pool3d_with_indices_backward(
                grad_output,
                input,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                indices,
            )

        return result, indices, grad_input

    cpu_input = torch.randn(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    fn_hpu = compile_function_if_compile_mode(fn)

    cpu_output, cpu_indices, cpu_grad = fn(cpu_input)
    hpu_output, hpu_indices, hpu_grad = fn_hpu(hpu_input)

    torch.testing.assert_close(hpu_output.cpu(), cpu_output)
    # Indices are calculated diffirently on HPU and CPU see: SW-220454
    # torch.testing.assert_close(hpu_indices.cpu(), cpu_indices)
    torch.testing.assert_close(hpu_grad.cpu(), cpu_grad)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"max_pool3d_with_indices", "max_pool3d_with_indices_backward"})
