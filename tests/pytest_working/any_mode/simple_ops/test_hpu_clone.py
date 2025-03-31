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

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int, torch.int8]
fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
dtypes += fp8_dtypes


@pytest.mark.parametrize("memory_format", [None, torch.contiguous_format], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_clone(memory_format, dtype):
    def fn(input):
        result = input.clone(memory_format=memory_format)
        return result

    fn = compile_function_if_compile_mode(fn)

    input = torch.randn((3, 4, 5)).to(dtype)
    input_hpu = input.to("hpu")

    result_cpu = input.clone(memory_format=memory_format)
    result_hpu = fn(input_hpu)

    input_hpu = input_hpu * 2

    tol = 1e-5
    compare_tensors(result_hpu, result_cpu, atol=tol, rtol=tol)
    assert not torch.equal(input_hpu, result_hpu)
    assert result_hpu.dtype == dtype

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("clone")
