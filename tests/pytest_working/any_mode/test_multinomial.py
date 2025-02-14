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

import numpy as np
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    is_gaudi1,
    is_pytest_mode_compile,
)

multinomial_dtypes = [torch.float, torch.bfloat16]
if not is_gaudi1():
    multinomial_dtypes.append(torch.float16)


@pytest.mark.parametrize("size", [(10,), (8, 8)])
@pytest.mark.parametrize("dtype", multinomial_dtypes)
@pytest.mark.parametrize("replacement", [True, False])
def test_multinomial(size, dtype, replacement):
    op = compile_function_if_compile_mode(torch.multinomial)

    input = torch.rand(size, dtype=dtype).to("hpu")
    result = op(input, 3, replacement=replacement).cpu()

    assert torch.all(result >= 0)
    assert torch.all(result < size[-1])

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("habana_multinomial")


@pytest.mark.parametrize("dtype", multinomial_dtypes)
def test_multinomial_output(dtype):
    op = compile_function_if_compile_mode(torch.multinomial)

    N = 1000
    input = torch.rand((10,), dtype=dtype).to("hpu")
    result = op(input, N, replacement=True).cpu()

    input = input.cpu()
    original_prob = input / torch.sum(input)
    result_prob = torch.bincount(result) / N

    diff = torch.abs(result_prob - original_prob)
    standard_error = torch.sqrt((original_prob * (1 - original_prob)) / N)

    assert np.alltrue((3 * standard_error > diff).numpy())

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("habana_multinomial")


@pytest.mark.parametrize("size", [(10,), (8, 8)])
@pytest.mark.parametrize("dtype", multinomial_dtypes)
@pytest.mark.parametrize("replacement", [True, False])
def test_multinomial_multiple_calls(size, dtype, replacement):
    op = compile_function_if_compile_mode(torch.multinomial)

    input = torch.rand(size, dtype=dtype).to("hpu")
    result = op(input, 4, replacement=replacement).cpu()
    next_result = op(input, 4, replacement=replacement).cpu()

    assert not torch.all(result == next_result), "Two consequtive multinomial op call should give different results."

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("habana_multinomial")
