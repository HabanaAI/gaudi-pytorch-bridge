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

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_convolution(dtype):
    def fn(input, weight, bias):
        return torch.convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

    minibatch, in_channels, out_channels, groups, W = 4, 8, 6, 2, 2
    stride = padding = dilation = output_padding = (1,)
    transposed=False
    input_shape = (minibatch, in_channels, W)
    weight_shape = (out_channels, int(in_channels/groups), W)
    bias_shape = (out_channels,)

    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_weight = torch.rand(weight_shape, dtype=dtype)
    hpu_weight = cpu_weight.to("hpu")
    cpu_bias = torch.rand(bias_shape, dtype=dtype)
    hpu_bias = cpu_bias.to("hpu")
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = cpu_compiled_fn(cpu_input, cpu_weight, cpu_bias)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_weight, hpu_bias)

    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output.cpu(), rtol=rtol)