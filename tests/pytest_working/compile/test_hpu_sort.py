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


@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("descending", [True, False])
def test_sort(dim, descending):
    def fn(input, dim, descending):
        return torch.sort(input, dim, descending)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    hx = x.to("hpu")

    result1, result2 = fn(x, dim, descending)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult1, hresult2 = compiled_fn(hx, dim, descending)

    assert torch.allclose(result1, hresult1.cpu(), atol=0.001, rtol=0.001)
    assert torch.allclose(result2, hresult2.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
def test_sort_stable(dim, descending, stable):
    def fn(input, dim, descending, stable):
        return input.sort(dim=dim, descending=descending, stable=stable)

    if dim in [3, -1] and not descending and stable:
        pytest.skip("SortStableFallbackCheck returns false")

    # CPU
    x = torch.randn([12, 10, 8, 6])
    hx = x.to("hpu")

    result1, result2 = fn(x, dim, descending, stable)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult1, hresult2 = compiled_fn(hx, dim, descending, stable)

    assert torch.allclose(result1, hresult1.cpu(), atol=0.001, rtol=0.001)
    assert torch.allclose(result2, hresult2.cpu(), atol=0.001, rtol=0.001)


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("descending", [False])
@pytest.mark.parametrize("stable", [True])
def test_sort_stable_bf16(dim, descending, stable):
    def fn(input, dim, descending, stable):
        return input.sort(dim=dim, descending=descending, stable=stable)

    # CPU
    x = torch.randn([8, 24, 24, 3], dtype=torch.bfloat16)
    hx = x.to("hpu")

    result1, result2 = fn(x, dim, descending, stable)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult1, hresult2 = compiled_fn(hx, dim, descending, stable)

    assert torch.allclose(result1, hresult1.cpu(), atol=0.001, rtol=0.001)
    assert torch.allclose(result2, hresult2.cpu(), atol=0.001, rtol=0.001)
