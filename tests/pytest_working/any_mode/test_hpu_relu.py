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
    is_pytest_mode_lazy,
)


@pytest.mark.skipif(
    is_pytest_mode_lazy(), reason="Error when trying to cast Long to Int, Input values range exceeds Int range"
)
def test_relu_long_large_number():
    cpu_tensor = torch.tensor([-1.0, 1.0, 2.0, -3.0, 1e18, -1e18], dtype=torch.long)
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = compile_function_if_compile_mode(torch.relu)(hpu_tensor)
    result_cpu = torch.relu(cpu_tensor)

    torch.testing.assert_close(result_hpu.cpu(), result_cpu)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("relu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int, torch.long], ids=format_tc)
def test_relu(dtype):
    cpu_tensor = torch.tensor([-1.0, 1.0, 2.0, -3.0], dtype=dtype)
    hpu_tensor = cpu_tensor.to("hpu")

    result_hpu = compile_function_if_compile_mode(torch.relu)(hpu_tensor)
    result_cpu = torch.relu(cpu_tensor)

    torch.testing.assert_close(result_hpu.cpu(), result_cpu)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("relu")
