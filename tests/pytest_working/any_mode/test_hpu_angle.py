###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

dtypes = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
inputs = [
    torch.tensor([0.0, 1.0, -1.0, float("nan")]),
    torch.tensor([-3.0, float("nan"), 0.0, 3.0]),
]


@pytest.mark.parametrize("input", inputs, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_angle(input, dtype):
    cpu_input = input.to(dtype)
    hpu_input = cpu_input.to("hpu")

    fn = torch.angle
    cpu_output = fn(cpu_input)
    fn = compile_function_if_compile_mode(fn)
    hpu_output = fn(hpu_input)

    compare_tensors(hpu_output, cpu_output, rtol=1e-7, atol=1e-7)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("angle")
