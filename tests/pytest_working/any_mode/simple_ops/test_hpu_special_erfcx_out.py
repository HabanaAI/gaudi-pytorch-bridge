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


@pytest.mark.parametrize("shape", [[2, 2]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_special_erfcx_out(shape, dtype):
    def fn(input, output):
        return torch.special.erfcx(input, out=output)

    cpu_input = torch.rand(shape, dtype=torch.float)
    hpu_input = cpu_input.to("hpu").to(dtype)
    cpu_output = torch.empty_like(cpu_input)
    hpu_output = torch.empty_like(hpu_input)
    fn(cpu_input, cpu_output)

    fn = compile_function_if_compile_mode(fn)

    fn(hpu_input, hpu_output)
    compare_tensors(cpu_output, hpu_output.cpu().to(torch.float), 0.005, 0.03)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("special_erfcx")
