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


@pytest.mark.parametrize("input_shape", [(8, 2, 4, 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_native_group_norm_bwd(input_shape, dtype):
    (N, C, H, W) = input_shape
    G = C
    weight_shape = C

    def fn(input, weight, bias):
        output = torch.native_group_norm(input, weight, bias, N, C, H * W, G, 0.1)
        grad = torch.ones_like(input)
        output[0].backward(grad)
        return input.grad

    cpu_input = torch.rand(input_shape, dtype=dtype, requires_grad=True)
    cpu_weight = torch.rand(weight_shape, dtype=dtype)
    cpu_bias = torch.rand(weight_shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu").detach().requires_grad_(True)
    hpu_weight = cpu_weight.to("hpu").detach()
    hpu_bias = cpu_bias.to("hpu").detach()

    cpu_results = fn(cpu_input, cpu_weight, cpu_bias)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    hpu_results = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)
    rtol = 1e-01 if dtype == torch.bfloat16 else 1e-03
    atol = 5e-02 if dtype == torch.bfloat16 else 1e-05

    assert torch.allclose(cpu_results, hpu_results.cpu(), rtol, atol)
