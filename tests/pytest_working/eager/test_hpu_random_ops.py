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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs


@pytest.mark.parametrize("n", [(5), (8), (17)])
@pytest.mark.parametrize("dtype", [torch.long])
def test_randperm(n, dtype):
    def fn(shape):
        return torch.randperm(shape, dtype=dtype, device="hpu")

    torch.manual_seed(1234)
    result_1 = fn(n).cpu()
    result_2 = fn(n).cpu()

    torch.manual_seed(1234)
    result_1a = fn(n).cpu()
    result_2a = fn(n).cpu()

    torch.manual_seed(12345)
    result_1b = fn(n).cpu()
    result_2b = fn(n).cpu()

    assert result_1.dtype == dtype
    assert not torch.equal(result_1, result_1b)
    assert not torch.equal(result_2, result_2b)

    assert torch.equal(result_1, result_1a)
    assert torch.equal(result_2, result_2a)

    # check_ops_executed_in_jit_ir("habana_randperm")
