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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize(
    "params",
    [
        ({"dims": (2, 3, 4, 5), "momentum": 0.999, "eps": 1e-5}),
        ({"dims": (2, 3, 4, 5, 6, 7), "momentum": 0.999, "eps": 1e-5}),
    ],
    ids=format_tc,
)
def test_hpu_native_batch_norm_legit_no_training(dtype, params):
    def fn(input, weight, bias, running_mean, running_var, momentum, eps):
        return torch._native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)

    torch._dynamo.reset()
    aot_hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    input = torch.randn(*params["dims"], dtype=dtype)
    weight = torch.randn(params["dims"][1])
    bias = torch.randn(params["dims"][1])
    running_mean = torch.randn(params["dims"][1])
    running_var = torch.randn(params["dims"][1])

    cpu_out = fn(input, weight, bias, running_mean, running_var, params["momentum"], params["eps"])
    hpu_out = aot_hpu_compiled_fn(
        input.to("hpu"),
        weight.to("hpu"),
        bias.to("hpu"),
        running_mean.to("hpu"),
        running_var.to("hpu"),
        params["momentum"],
        params["eps"],
    )

    assert torch.allclose(cpu_out[0], hpu_out[0].to("cpu"), equal_nan=True)


@pytest.mark.parametrize("shape", [[4, 3, 8]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_native_batch_norm_bwd(shape, dtype):
    def fn(input, weight, bias, running_mean, running_var):
        native_batch_norm = torch.native_batch_norm(
            input,
            weight,
            bias,
            running_mean=running_mean,
            running_var=running_var,
            training=True,
            momentum=0.1,
            eps=1e-5,
        )
        grad = torch.ones_like(native_batch_norm[0])
        native_batch_norm[0].backward(grad)
        return input.grad

    num_channels = shape[1]
    cpu_input = torch.randn(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_input.requires_grad = True
    hpu_input.requires_grad = True
    cpu_weight = torch.ones(num_channels, dtype=dtype)
    hpu_weight = cpu_weight.to("hpu")
    cpu_bias = torch.zeros(num_channels, dtype=dtype)
    hpu_bias = cpu_bias.to("hpu")
    cpu_running_mean = torch.zeros(num_channels, dtype=dtype)
    hpu_running_mean = cpu_running_mean.to("hpu")
    cpu_running_var = torch.ones(num_channels, dtype=dtype)
    hpu_running_var = cpu_running_var.to("hpu")
    torch._dynamo.reset()

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, cpu_weight, cpu_bias, cpu_running_mean, cpu_running_var)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias, hpu_running_mean, hpu_running_var).cpu()
    tol = 1e-3 if dtype == torch.bfloat16 else 1e-6
    assert torch.allclose(cpu_output, hpu_output, atol=tol)
