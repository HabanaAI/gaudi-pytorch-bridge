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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch


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
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx, dim)
    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)
