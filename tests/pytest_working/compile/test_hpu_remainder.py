###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
