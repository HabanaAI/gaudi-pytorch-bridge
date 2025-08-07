###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize(
    "input_size, output_size",
    [[[1, 12, 12], [5, 7]], [[2, 4, 12, 12], [4, 4]], [[4, 12, 12, 12], [5, 7, 6]], [[2, 4, 12, 12, 12], [4, 4, 6]]],
    ids=format_tc,
)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.float16], ids=format_tc)
def test_hpu_adaptive_max_pool(input_size, output_size, dtype):
    input_cpu = torch.randn(input_size, dtype=dtype)
    input_hpu = input_cpu.to("hpu")

    test_2d = len(output_size) == 2

    def fn(input):
        op = torch.nn.AdaptiveMaxPool2d(output_size) if test_2d else torch.nn.AdaptiveMaxPool3d(output_size)
        return op(input)

    result_cpu = fn(input_cpu)
    result_hpu = compile_function_if_compile_mode(fn)(input_hpu)

    torch.testing.assert_close(result_hpu.cpu(), result_cpu)

    if is_pytest_mode_compile():
        executed_ops = {"adaptive_max_pool2d"} if test_2d else {"adaptive_max_pool3d"}
        check_ops_executed_in_jit_ir(executed_ops)
