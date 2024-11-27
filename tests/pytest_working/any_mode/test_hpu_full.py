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

import os

import habana_frameworks.torch.internal.bridge_config as bc
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

test_data = [
    (torch.bool, True),
    (torch.bool, False),
    (torch.float, 2.5),
    (torch.bfloat16, 2.5),
    (torch.int16, 42),
    (torch.int32, 42.5),
    (torch.float, 42),
    (torch.int8, 42),
    (torch.uint8, 42),
    (torch.long, 42),
    (None, 30522.5),
    (None, 30522),
    (torch.int32, 30522),
    (torch.int16, 30522),
    (torch.float, 30522),
    (torch.int64, -42),
    (torch.int64, 123456789123456789),
    (torch.int64, -123456789123456789),
]

if not is_gaudi1():
    test_data += [
        (torch.float8_e5m2, 16.0),
        (torch.float8_e4m3fn, 16.0),
        (torch.float16, 42.0),
    ]


@pytest.mark.parametrize("size", [(1,), (1, 1), (2, 3)], ids=format_tc)
@pytest.mark.parametrize("dtype, fill_value", test_data, ids=format_tc)
def test_full(size, dtype, fill_value):
    if abs(fill_value) > 0x7FFFFFFF and bc.get_pt_enable_int64_support() == False:
        pytest.skip(reason="fill_value exceed int32 range which is unsupported")

    if is_pytest_mode_compile() and dtype == torch.bool:
        pytest.skip(reason="For bool input fallback to eager is expected in compile mode")

    def fn(size, fill_value, device, dtype):
        return torch.full(size, fill_value, device=device, dtype=dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(size, fill_value=fill_value, dtype=dtype, device="hpu")
    resultType = result.dtype

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        dtype = torch.float
        resultType = torch.float
    expected = torch.full(size, fill_value=fill_value, dtype=dtype, device="cpu")
    expectedType = expected.dtype

    compare_tensors([result], [expected], atol=0, rtol=0)
    assert resultType == expectedType
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("full")
