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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import numpy as np

@pytest.mark.parametrize("shape", [(1, 3, 4, 4)])
@pytest.mark.parametrize("eps", [0.01, 0.1])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_native_layer_norm(shape, eps, dtype):
    def fn(input, weight, bias):
        return torch.native_layer_norm(input, shape, weight, bias, eps)

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
    cpu_compiled_fn = torch.compile(fn)

    extended_shape = (10,) + shape
    cpu_input = torch.rand(extended_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_weight = torch.rand(shape, dtype=dtype)
    hpu_weight = cpu_weight.to("hpu")
    cpu_bias = torch.full(shape, 1.0, dtype=dtype)
    hpu_bias = cpu_bias.to("hpu")

    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)
    cpu_results = cpu_compiled_fn(cpu_input, cpu_weight, cpu_bias)
    assert torch.allclose(cpu_results[0], hpu_results[0].cpu(), 1e-03)

@pytest.mark.parametrize("shape", [(1, 3, 4, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_native_layer_norm_bwd(shape, dtype):
    extended_shape = (10,) + shape
    def fn(input, weight, bias):
        input.requires_grad = True
        output = torch.native_layer_norm(input, shape, weight, bias, 0.1)
        grad = torch.ones_like(input)
        output[0].backward(grad)
        return input.grad

    cpu_input = torch.rand(extended_shape, dtype=dtype)
    cpu_weight = torch.rand(shape, dtype=dtype)
    cpu_bias = torch.full(shape, 1.0, dtype=dtype)

    hpu_input = cpu_input.to("hpu")
    hpu_weight = cpu_weight.to("hpu")
    hpu_bias = cpu_bias.to("hpu")

    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    cpu_results = cpu_compiled_fn(cpu_input, cpu_weight, cpu_bias)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)
    rtol = 1e-01 if dtype == torch.bfloat16 else 1e-03
    assert torch.allclose(cpu_results[0], hpu_results[0].cpu(), rtol)
    assert torch.allclose(cpu_results[1], hpu_results[1].cpu(), rtol)
    assert torch.allclose(cpu_results[2], hpu_results[2].cpu(), rtol)
