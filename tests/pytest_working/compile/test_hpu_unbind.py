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


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(12, 10, 8, 6)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int8, torch.int32])
def test_unbind(dim, shape, dtype):
    def fn(input, dim):
        return torch.unbind(input, dim)

    # CPU
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype)
    else:
        x = torch.randint(low=-128, high=127, size=shape, dtype=dtype)

    hx = x.to("hpu")

    result = fn(x, dim)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx, dim)

    for a, b in zip(result, hresult):
        assert torch.allclose(a, b.cpu(), atol=0.001, rtol=0.001)
