###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
