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

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("size", [(1,), (2, 3)])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("op", [torch.zeros, torch.ones])
def test_op(size, dtype, op):
    def fn(size, dtype, device):
        return op(size, dtype=dtype, device=device)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(size, dtype=dtype, device="hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        dtype = torch.float
    expected = fn(size, dtype=dtype, device="cpu")

    compare_tensors([result], [expected], atol=0, rtol=0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("full")
