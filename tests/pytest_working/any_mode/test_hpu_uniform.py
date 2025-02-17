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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, format_tc, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.float, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float16]


@pytest.mark.parametrize("shape", [(), (48,), (24, 48), (1, 2, 3, 4, 5, 6, 7)], ids=format_tc)
@pytest.mark.parametrize("low, high", [(20, 120), (-50, 10)])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_uniform(shape, low, high, dtype):
    def fn(input, low, high):
        input.uniform_(low, high)

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        clear_t_compile_logs()
        fn = torch.compile(fn, backend="hpu_backend")

    input = torch.empty(shape, dtype=dtype, device="hpu")

    fn(input, low, high)
    res1 = input.cpu()

    fn(input, low, high)
    res2 = input.cpu()

    assert res1.shape == shape

    assert not torch.equal(res1, res2)
    assert torch.all(res1 <= high) and torch.all(res1 >= low)
    assert torch.all(res2 <= high) and torch.all(res2 >= low)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("habana_uniform")
