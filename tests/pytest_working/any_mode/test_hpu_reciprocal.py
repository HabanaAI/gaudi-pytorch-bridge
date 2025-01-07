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
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16]
fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
if not is_gaudi1():
    dtypes += fp8_dtypes


@pytest.mark.parametrize("shape", [(), (1,), (20, 40)])
@pytest.mark.parametrize("dtype", dtypes)
def test_reciprocal(shape, dtype):
    def fn(input):
        return torch.reciprocal(input)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    input = (torch.randn(shape) * 10.0).to(dtype)

    input_hpu = input.to("hpu")
    if dtype in fp8_dtypes:
        input = input.float()

    expected = torch.reciprocal(input).to(dtype)
    result = fn(input_hpu)

    tol = 1e-5
    compare_tensors(result, expected, atol=tol, rtol=tol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("reciprocal")
