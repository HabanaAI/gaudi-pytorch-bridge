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
import os

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int, torch.bool]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("op", [torch.amin, torch.amax, torch.aminmax], ids=format_tc)
@pytest.mark.parametrize("shape", [(3, 4, 5, 6)], ids=format_tc)
@pytest.mark.parametrize("dim", [None, -4, -3, -2, -1, 0, 1, 2, 3], ids=format_tc)
@pytest.mark.parametrize("keepdim", [False, True], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_amax_amin(op, shape, dim, keepdim, dtype):
    def fn(input, dim, keepdim):
        return op(input, dim=dim, keepdim=keepdim)

    if dtype == torch.int:
        cpu_input = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
    elif dtype == torch.bool:
        cpu_input = torch.randint(low=0, high=1, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        cpu_input = cpu_input.float()

    cpu_output = fn(cpu_input, dim, keepdim)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    hpu_output = fn(hpu_input, dim, keepdim)

    if is_pytest_mode_compile():
        ops_expected = op.__name__ if op != torch.aminmax else {"amin", "amax"}
        check_ops_executed_in_jit_ir(ops_expected)

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)
