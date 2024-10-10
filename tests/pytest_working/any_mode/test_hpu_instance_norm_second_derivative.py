###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
from test_utils import format_tc, is_pytest_mode_compile


@pytest.mark.parametrize("shape", [(4, 2, 3, 1), (10, 3, 4), (10, 4, 2, 3, 3)], ids=format_tc)
@pytest.mark.parametrize("use_weight_and_bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.skipif(
    is_pytest_mode_compile(),
    reason="RuntimeError: torch.compile with aot_autograd does not currently support double backward",
)
def test_hpu_instance_norm_second_derivative(shape, use_weight_and_bias, dtype):
    channel_size = shape[1]
    input_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
    weight_cpu, bias_cpu = (
        (
            torch.randn(channel_size, dtype=dtype, requires_grad=True),
            torch.randn(channel_size, dtype=dtype, requires_grad=True),
        )
        if use_weight_and_bias
        else (None, None)
    )

    input_hpu = input_cpu.to("hpu").detach().requires_grad_(True)
    weight_hpu, bias_hpu = (
        (weight_cpu.to("hpu").detach().requires_grad_(True), bias_cpu.to("hpu").detach().requires_grad_(True))
        if use_weight_and_bias
        else (None, None)
    )

    def fn(input, weight, bias):
        res = torch.nn.functional.instance_norm(input, running_mean=None, running_var=None, weight=weight, bias=bias)
        first_derivatives = torch.autograd.grad(
            inputs=(input, weight, bias) if use_weight_and_bias else (input),
            outputs=res,
            grad_outputs=torch.ones_like(res),
            retain_graph=True,
            create_graph=True,
        )
        # bias is not an input tensor to the calculation of second derivatvie
        second_derivatives = torch.autograd.grad(
            inputs=(input, weight) if use_weight_and_bias else (input),
            outputs=first_derivatives,
            grad_outputs=[torch.ones_like(deriv) for deriv in first_derivatives],
        )
        return (second_derivatives[0], second_derivatives[1]) if use_weight_and_bias else (second_derivatives[0],)

    results_cpu = fn(input_cpu, weight_cpu, bias_cpu)
    results_hpu = fn(input_hpu, weight_hpu, bias_hpu)

    tol = 7e-2 if dtype == torch.bfloat16 else 1e-5
    for res_cpu, res_hpu in zip(results_cpu, results_hpu):
        assert torch.allclose(res_cpu, res_hpu.cpu(), atol=tol, rtol=tol)
