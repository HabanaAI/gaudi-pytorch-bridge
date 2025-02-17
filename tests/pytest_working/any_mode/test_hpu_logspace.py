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
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("start", [0, 1, 6])
@pytest.mark.parametrize("end", [5, 8, 12])
@pytest.mark.parametrize("steps", [0, 1, 6, 13])
@pytest.mark.parametrize("base", [1, 2, 10])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_logspace_float(start, end, steps, base, dtype):
    def fn(start, end, steps, base, dtype, device="cpu"):
        return torch.logspace(start, end, steps, base=float(base), dtype=dtype, device=device)

    cpu_output = fn(start, end, steps, base, dtype)
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_fn = torch.compile(fn, backend="hpu_backend")
    else:
        hpu_fn = fn

    hpu_output = hpu_fn(start, end, steps, base, dtype, device="hpu")

    if dtype == torch.bfloat16 and (start >= 5 or end >= 5):
        rtol = 2e-1
    else:
        rtol = 1e-5

    compare_tensors([hpu_output], [cpu_output], atol=1e-8, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("logspace")


@pytest.mark.parametrize("start", [0, 1, 6])
@pytest.mark.parametrize("end", [5, 8, 0])
@pytest.mark.parametrize("steps", [0, 1, 6, 13, 100, 1000])
@pytest.mark.parametrize("base", [1, 2, 10])
@pytest.mark.parametrize("dtype", [torch.int32], ids=format_tc)
def test_hpu_logspace_int(start, end, steps, base, dtype):
    def fn(start, end, steps, base, dtype, device="cpu"):
        return torch.logspace(start, end, steps, base=base, dtype=dtype, device=device)

    cpu_output = fn(start, end, steps, base, dtype)
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_fn = torch.compile(fn, backend="hpu_backend")
    else:
        hpu_fn = fn

    hpu_output = hpu_fn(start, end, steps, base, dtype, device="hpu")

    compare_tensors([hpu_output], [cpu_output], atol=1, rtol=1e-3)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("logspace")
