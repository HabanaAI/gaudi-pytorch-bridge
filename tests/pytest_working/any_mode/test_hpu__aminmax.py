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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.bool]
integer_dtypes = [torch.int, torch.long]
if not is_gaudi1():
    dtypes += [torch.half, torch.float8_e5m2, torch.float8_e4m3fn]


def prepare_input(shape, dtype):
    if dtype in integer_dtypes:
        low = -100
        high = 100
        cpu_input = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    elif dtype is torch.bool:
        cpu_input = torch.randint(low=0, high=1, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        cpu_input = cpu_input.float()
    return cpu_input, hpu_input


def run_test(shape, dtype, fn, *args):
    cpu_input, hpu_input = prepare_input(shape, dtype)

    cpu_output = fn(cpu_input, *args)

    fn = compile_function_if_compile_mode(fn)

    hpu_output = fn(hpu_input, *args)

    compare_tensors(hpu_output, cpu_output, rtol=1e-05, atol=1e-05)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_aminmax")


@pytest.mark.parametrize("shape", [(3, 4, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu__aminmax(shape, dtype):
    def fn(input):
        return torch._aminmax(input)

    run_test(shape, dtype, fn)


@pytest.mark.parametrize("shape", [(3, 4, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
@pytest.mark.parametrize("dim", [-2, -1, 0, 1], ids=format_tc)
@pytest.mark.parametrize("keepdim", [False, True], ids=format_tc)
def test_hpu__aminmax_dim_keepdim(shape, dtype, dim, keepdim):
    def fn(input, dim, keepdim):
        return torch._aminmax(input, dim, keepdim)

    run_test(shape, dtype, fn, dim, keepdim)
