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


def test_remainder_tensor():
    def fn(input, other):
        return torch.remainder(input, other)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    y = torch.randn([12, 10, 8, 6])
    hx = x.to("hpu")
    hy = y.to("hpu")

    result = fn(x, y)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx, hy)
    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)


def test_remainder_scalar():
    def fn(input, other):
        return torch.remainder(input, other)

    # CPU
    x = torch.randn([12, 10, 8, 6])
    y = 5.0
    hx = x.to("hpu")

    result = fn(x, y)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(hx, y)
    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)


def test_remainder_scalar_tensor():
    def fn(input, other):
        return torch.remainder(input, other)

    # CPU
    x = 5.0
    y = torch.randn([12, 10, 8, 6])
    hy = y.to("hpu")

    result = fn(x, y)

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")

    hresult = compiled_fn(x, hy)
    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)
