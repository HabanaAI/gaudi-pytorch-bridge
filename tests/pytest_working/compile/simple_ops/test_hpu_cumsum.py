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
from test_utils import check_ops_executed_in_jit_ir, compile_function_if_compile_mode


@pytest.mark.parametrize("op_code", [torch.cumsum])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
def test_cumsum_dim(op_code, dim):
    def fn(input, dim):
        return op_code(input, dim)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    hx = x.to("hpu")

    result = fn(x, dim)

    # HPU
    compiled_fn = compile_function_if_compile_mode(fn)

    hresult = compiled_fn(hx, dim)

    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)
    check_ops_executed_in_jit_ir("cumsum")
