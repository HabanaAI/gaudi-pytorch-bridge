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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch

shapes = [(3, 1, 7, 4, 1), (1, 5, 1, 1, 8)]
dims = [(0, 3), (-1, 2), (1, -2, 0)]
dim = [0, 1, 4]


@pytest.mark.parametrize("shape", shapes)
def test_hpu_squeeze(shape):
    def fn(shape):
        return torch.squeeze(shape).relu()

    torch._dynamo.reset()
    shape = torch.zeros(shape).to("hpu")
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    expected = fn(shape).to("cpu")
    result = compiled_fn(shape).to("hpu").to("cpu")
    assert torch.equal(result, expected)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dim", dim)
def test_hpu_squeeze_dim(shape, dim):
    def fn(shape, dim):
        return torch.squeeze(shape, dim).relu()

    torch._dynamo.reset()
    shape = torch.zeros(shape).to("hpu")
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    expected = fn(shape, dim).to("cpu")
    result = compiled_fn(shape, dim).to("hpu").to("cpu")
    assert torch.equal(result, expected)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dims", dims)
def test_hpu_squeeze_dims(shape, dims):
    def fn(shape, dims):
        return torch.squeeze(shape, dims).relu()

    torch._dynamo.reset()
    shape = torch.zeros(shape).to("hpu")
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    expected = fn(shape, dims).to("cpu")
    result = compiled_fn(shape, dims).to("hpu").to("cpu")
    assert torch.equal(result, expected)


@pytest.mark.parametrize("shape", [(1,), (4,)])
@pytest.mark.parametrize("dim", [0, (0,)])
def test_hpu_squeeze_dim0(shape, dim):
    def fn(input, dims):
        return torch.squeeze(input, dims).relu()

    torch._dynamo.reset()
    input = torch.randn(shape) * 5.0
    input_hpu = input.to("hpu")

    compiled_fn = torch.compile(fn, backend="hpu_backend")
    result_cpu = fn(input, dim)
    result_hpu = compiled_fn(input_hpu, dim)

    assert torch.equal(result_hpu.cpu(), result_cpu)
