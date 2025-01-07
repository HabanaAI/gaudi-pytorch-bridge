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
from test_utils import format_tc


@pytest.mark.parametrize("shape", [(1, 3, 4, 4)], ids=format_tc)
@pytest.mark.parametrize("eps", [0.01, 0.1])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_native_layer_norm(shape, eps, dtype):
    def fn(input, weight, bias):
        return torch.native_layer_norm(input, shape, weight, bias, eps)

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    extended_shape = (10,) + shape
    cpu_input = torch.rand(extended_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_weight = torch.rand(shape, dtype=dtype)
    hpu_weight = cpu_weight.to("hpu")
    cpu_bias = torch.full(shape, 1.0, dtype=dtype)
    hpu_bias = cpu_bias.to("hpu")

    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)
    cpu_results = fn(cpu_input, cpu_weight, cpu_bias)
    assert torch.allclose(cpu_results[0], hpu_results[0].cpu(), 1e-03)


@pytest.mark.parametrize("shape", [(1, 3, 4, 4)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_native_layer_norm_bwd(shape, dtype):
    extended_shape = (10,) + shape

    def fn(input, weight, bias):
        output = torch.native_layer_norm(input, shape, weight, bias, 0.001)
        grad = torch.ones_like(input)
        output[0].backward(grad)
        return input.grad

    cpu_input = torch.rand(extended_shape, dtype=dtype, requires_grad=True)
    cpu_weight = torch.rand(shape, dtype=dtype)
    cpu_bias = torch.full(shape, 1.0, dtype=dtype)

    hpu_input = cpu_input.to("hpu").detach().requires_grad_(True)
    hpu_weight = cpu_weight.to("hpu")
    hpu_bias = cpu_bias.to("hpu")

    torch._dynamo.reset()
    cpu_results = fn(cpu_input, cpu_weight, cpu_bias)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)
    rtol = 5e-02 if dtype == torch.bfloat16 else 1e-03
    atol = 5e-02 if dtype == torch.bfloat16 else 1e-05

    assert torch.allclose(cpu_results, hpu_results.cpu(), rtol, atol)
