###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import random

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    is_pytest_mode_compile,
)

supported_dtypes = [torch.float, torch.bfloat16, torch.long, torch.int, torch.short, torch.half]


def generate_inputs(shape, dtype):
    if dtype in [torch.float, torch.bfloat16, torch.half]:
        input = torch.randn(shape, dtype=dtype)
        other = random.uniform(-2.0, 2.0)
    else:
        input = torch.randint(0, 10, shape, dtype=dtype)
        other = random.randint(0, 10)

    return (input, other, input.to("hpu"))


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_hpu_add_scalar(dtype):
    input, other, input_hpu = generate_inputs((8, 12), dtype)

    def op(a, b):
        return torch.add(a, b)

    op = compile_function_if_compile_mode(op)

    result = op(input, other)
    result_hpu = op(input_hpu, other)

    assert torch.allclose(result_hpu.cpu(), result)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("add")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_hpu_add_scalar_inplace(dtype):

    input, other, input_hpu = generate_inputs((8, 12), dtype)

    def op(a, b):
        return a.add_(b)

    op = compile_function_if_compile_mode(op)

    op(input, other)
    op(input_hpu, other)

    assert torch.allclose(input_hpu.cpu(), input)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("add")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_hpu_add_scalar_out(dtype):

    input, other, input_hpu = generate_inputs((8, 12), dtype)
    out = torch.empty_like(input)
    out_hpu = torch.empty_like(input_hpu)

    def op(a, b, out):
        return torch.add(a, b, out=out)

    op = compile_function_if_compile_mode(op)

    op(input, other, out)
    op(input_hpu, other, out_hpu)

    assert torch.allclose(out_hpu.cpu(), out)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("add")
