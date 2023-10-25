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
from test_utils import is_torch_at_least


@pytest.mark.parametrize("input_shape", [(2, 2, 2, 2)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_native_layer_norm_bwd(input_shape, dtype):
    if is_torch_at_least(2, 1):
        pytest.xfail("https://jira.habana-labs.com/browse/SW-161574")

    (N, C, H, W) = input_shape
    G = C
    weight_shape = C

    def fn(input, weight, bias):
        input.requires_grad = True
        output = torch.native_group_norm(input, weight, bias, N, C, H * W, G, 0.1)
        grad = torch.ones_like(input)
        output[0].backward(grad)
        return input.grad

    cpu_input = torch.rand(input_shape, dtype=dtype)
    cpu_weight = torch.rand(weight_shape, dtype=dtype)
    cpu_bias = torch.full(input_shape, 1.0, dtype=dtype)

    hpu_input = cpu_input.to("hpu")
    hpu_weight = cpu_weight.to("hpu")
    hpu_bias = cpu_bias.to("hpu")

    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    cpu_results = cpu_compiled_fn(cpu_input, cpu_weight, cpu_bias)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)
    rtol = 1e-01 if dtype == torch.bfloat16 else 1e-03
    for cpu_result, hpu_result in zip(cpu_results, hpu_results):
        assert torch.allclose(cpu_result, hpu_result.cpu(), rtol)
