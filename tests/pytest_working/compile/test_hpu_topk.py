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


@pytest.mark.parametrize("dtype", [None, torch.float, torch.bfloat16])
@pytest.mark.parametrize("k", [4, 6])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("sorted", [True])
def test_topk(k, dim, largest, sorted, dtype):
    def fn(input, k, dim, largest, sorted):
        return torch.topk(input, k, dim, largest, sorted)

    # CPU
    x = torch.randn([12, 10, 8, 6], dtype=dtype)
    hx = x.to("hpu")

    result1, result2 = fn(x, k, dim, largest, sorted)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult1, hresult2 = compiled_fn(hx, k, dim, largest, sorted)

    assert torch.allclose(result1, hresult1.cpu(), atol=0.001, rtol=0.001)
