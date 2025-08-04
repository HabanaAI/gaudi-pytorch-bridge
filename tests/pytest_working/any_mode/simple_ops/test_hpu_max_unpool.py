###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    filter_dtypes,
    format_tc,
    is_pytest_mode_compile,
)

dtypes = filter_dtypes([torch.float32, torch.bfloat16, torch.float16])


@pytest.mark.parametrize("shape", [[3, 2, 16, 16], [8, 13, 14]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("kernel_size", [2, (2, 4)], ids=format_tc)
@pytest.mark.parametrize("stride", [1, (2, 4)], ids=format_tc)
@pytest.mark.parametrize("padding", [(0, 1)], ids=format_tc)
def test_hpu_max_unpool2d(shape, dtype, kernel_size, stride, padding):
    cpu_tensor = torch.randn(shape).to(dtype)
    cpu_output, cpu_indices = torch.nn.functional.max_pool2d(
        cpu_tensor, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True
    )
    hpu_output, hpu_indices = cpu_output.to("hpu"), cpu_indices.to("hpu")

    def fn(input, indices, kernel_size, stride, padding, output_size):
        return torch.nn.functional.max_unpool2d(
            input, indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=output_size
        )

    hpu_fn = compile_function_if_compile_mode(fn)

    cpu_result = fn(
        cpu_output, cpu_indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=shape[-2:]
    )
    hpu_result = hpu_fn(
        hpu_output, hpu_indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=shape[-2:]
    )

    torch.testing.assert_close(hpu_result.cpu(), cpu_result, rtol=0, atol=0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("max_unpool2d")


@pytest.mark.parametrize("shape", [[1, 2, 16, 16, 16], [2, 12, 13, 14]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("kernel_size", [2, (2, 4, 2)], ids=format_tc)
@pytest.mark.parametrize("stride", [1, (2, 3, 2)], ids=format_tc)
@pytest.mark.parametrize("padding", [(0, 1, 0)], ids=format_tc)
def test_hpu_max_unpool3d(shape, dtype, kernel_size, stride, padding):
    cpu_tensor = torch.randn(shape).to(dtype)
    cpu_output, cpu_indices = torch.nn.functional.max_pool3d(
        cpu_tensor, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True
    )
    hpu_output, hpu_indices = cpu_output.to("hpu"), cpu_indices.to("hpu")

    def fn(input, indices, kernel_size, stride, padding, output_size):
        return torch.nn.functional.max_unpool3d(
            input, indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=output_size
        )

    hpu_fn = compile_function_if_compile_mode(fn)

    cpu_result = fn(
        cpu_output, cpu_indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=shape[-3:]
    )
    hpu_result = hpu_fn(
        hpu_output, hpu_indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=shape[-3:]
    )

    torch.testing.assert_close(hpu_result.cpu(), cpu_result, rtol=0, atol=0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("max_unpool3d")
