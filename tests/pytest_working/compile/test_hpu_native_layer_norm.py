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
