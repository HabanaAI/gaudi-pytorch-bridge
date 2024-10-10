###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
def test_hpu_convolution(dtype):
    def fn(input, weight, bias):
        return torch.convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

    minibatch, in_channels, out_channels, groups, W = 4, 8, 6, 2, 2
    stride = padding = dilation = output_padding = (1,)
    transposed = False
    input_shape = (minibatch, in_channels, W)
    weight_shape = (out_channels, int(in_channels / groups), W)
    bias_shape = (out_channels,)

    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_weight = torch.rand(weight_shape, dtype=dtype)
    hpu_weight = cpu_weight.to("hpu")
    cpu_bias = torch.rand(bias_shape, dtype=dtype)
    hpu_bias = cpu_bias.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, cpu_weight, cpu_bias)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)

    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output.cpu(), rtol=rtol)


def test_hpu_convolution_grad_with_view():
    def run(device):
        m = torch.nn.Conv2d(5, 6, (2, 2), stride=(1, 1), bias=False).to(device)
        m.weight = torch.nn.Parameter(torch.arange(1.0 * 120).reshape([30, 2, 2]).to(device).view([6, 5, 2, 2]))

        def fn(x):
            x = x.view([2, 5, 3, 4])
            return m(x)

        if device == "hpu":
            backend = "hpu_backend"
            fn = torch.compile(fn, backend=backend)

        x = torch.arange(1.0 * 120).reshape([10, 3, 4]).to(device)
        x.requires_grad_()

        res = fn(x)

        grad_in = torch.ones(2, 36).to(device).view(2, 6, 2, 3)
        res.backward(grad_in)

        res = [p.grad for p in m.parameters()]
        res.append(x.grad)
        return res

    cpu_grads = run("cpu")
    hpu_grads = run("hpu")

    for cpu_grad, hpu_grad in zip(cpu_grads, hpu_grads):
        assert torch.allclose(cpu_grad, hpu_grad.cpu())
