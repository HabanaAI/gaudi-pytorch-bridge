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
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
    is_pytest_mode_lazy,
)

dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
    torch.int8,
    torch.int16,
]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]

TEST_CASES = [((5, 7), -2, -5, 5), ((6, 4, 3), 0, 5, 1), ((2, 2), 0, 2, 0)]


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("shape, dim, start, length", TEST_CASES, ids=format_tc)
def test_hpu_narrow(shape, dim, start, length, dtype):
    if is_pytest_mode_lazy and dtype == torch.int64:
        pytest.skip("int64 not supported in lazy mode")
    if dtype in [torch.int32, torch.int64, torch.int8, torch.int16]:
        input = torch.randint(low=-5, high=5, size=shape, dtype=dtype)
    elif dtype == torch.uint8:
        input = torch.randint(low=0, high=5, size=shape, dtype=dtype)
    elif dtype == torch.bool:
        input = torch.zeros(shape, dtype=dtype)
        input[torch.randn(*shape) > 0.5] = True
    else:
        input = torch.randn(shape).to(dtype)
    input_h = input.to("hpu")

    def fn(self_tensor, dim, start, length):
        return torch.narrow_copy(self_tensor, dim, start, length)

    fn = compile_function_if_compile_mode(fn)

    output_h = fn(input_h, dim, start, length)
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        # Convert to float32 for comparison
        output_h = output_h.float()
        input = input.float()
    output = torch.narrow_copy(input, dim, start, length)

    compare_tensors(output_h, output, atol=0.0, rtol=0.0)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"narrow_copy"})
