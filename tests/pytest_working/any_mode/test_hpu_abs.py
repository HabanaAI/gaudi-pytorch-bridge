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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_lazy, is_pytest_mode_compile

dtypes = [torch.float32, torch.bfloat16]
integer_dtypes = [torch.int]
if not is_lazy():
    integer_dtypes += [torch.int16, torch.int8, torch.uint8]


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_abs(shape, dtype):
    def fn(input):
        return torch.abs(input)

    if dtype in integer_dtypes:
        low = 0 if dtype == torch.uint8 else -100
        high = 100
        cpu_input = torch.randint(low=low, high=high, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = torch.abs(cpu_input)
    hpu_output = fn(hpu_input).cpu()

    assert torch.equal(cpu_output, hpu_output)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("abs")
