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
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int8, torch.int16, torch.int, torch.int64, torch.uint8]
if not is_gaudi1():
    dtypes += [torch.float16, torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("from_dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("to_dtype", dtypes, ids=format_tc)
def test_hpu_cast(from_dtype, to_dtype):
    def fn(input):
        return input.to(to_dtype)

    to = 300 if from_dtype == torch.float and to_dtype in [torch.int8, torch.uint8] and not is_gaudi1() else 100
    input = torch.randint(0, to, (16, 16)).to(from_dtype)
    input_hpu = input.to("hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result_cpu = input.to(to_dtype)
    result_hpu = fn(input_hpu)

    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=0.0)

    if is_pytest_mode_compile() and from_dtype != to_dtype:
        check_ops_executed_in_jit_ir("_to_copy")
