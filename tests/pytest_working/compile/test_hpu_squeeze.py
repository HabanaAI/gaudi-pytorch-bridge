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
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend

shapes = [(3, 1, 7, 4, 1), (1, 5, 1, 1, 8)]
dims = [(0, 3), (-1, 2), (1, -2, 0)]
dim = [0, 1, 4]

@pytest.mark.parametrize("shape", shapes)
def test_hpu_squeeze(shape):
    def fn(shape):
        return torch.squeeze(shape).relu()

    shape = torch.zeros(shape).to("hpu")
    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
    expected = fn(shape).to("cpu")
    result = compiled_fn(shape).to("hpu").to("cpu")
    assert torch.equal(result, expected)

@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dim", dim)
def test_hpu_squeeze_dim(shape, dim):
    def fn(shape, dim):
        return torch.squeeze(shape, dim).relu()

    shape = torch.zeros(shape).to("hpu")
    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
    expected = fn(shape, dim).to("cpu")
    result = compiled_fn(shape, dim).to("hpu").to("cpu")
    assert torch.equal(result, expected)

@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dims", dims)
def test_hpu_squeeze_dims(shape, dims):
    def fn(shape, dims):
        return torch.squeeze(shape, dims).relu()

    shape = torch.zeros(shape).to("hpu")
    compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
    expected = fn(shape, dims).to("cpu")
    result = compiled_fn(shape, dims).to("hpu").to("cpu")
    assert torch.equal(result, expected)