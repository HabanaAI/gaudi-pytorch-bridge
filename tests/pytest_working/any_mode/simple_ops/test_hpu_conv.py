###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch
from test_utils import compile_function_if_compile_mode, cpu, format_tc, hpu, is_lazy
from torch import nn

Verbose = False


# N - batch
# H - input height
# W - input width
# C - input channels
# R - filter height
# S - filter width
# K - output channels
# str - stride
# pad - padding
# bias
mnist_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, bias
    (8, 28, 28, 1, 5, 5, 20, 1, 0, True),
    (8, 11, 11, 20, 5, 5, 50, 1, 0, True),
]

resnet50_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, bias
    pytest.param(
        64,
        224,
        224,
        3,
        7,
        7,
        64,
        2,
        3,
        False,
        marks=[pytest.mark.skip(reason="Too long test, simulator timeout")],
    ),
    pytest.param(
        64,
        56,
        56,
        64,
        3,
        3,
        64,
        1,
        1,
        False,
        marks=[pytest.mark.skip(reason="Too long test, simulator timeout")],
    ),
    pytest.param(
        64,
        56,
        56,
        128,
        3,
        3,
        128,
        2,
        1,
        False,
        marks=[pytest.mark.skip(reason="Too long test, simulator timeout")],
    ),
]


conv_chlast_test_case_list = (
    [
        # N, H, W, C, R, S, K, str, pad, bias
        (2, 3, 4, 5, 2, 2, 6, 1, 0, True),
        (4, 28, 28, 3, 2, 2, 16, 1, 0, True),
        (3, 28, 28, 3, 2, 2, 16, 1, 1, False),
    ]
    + mnist_test_case_list
    + resnet50_test_case_list
)

conv_bwd_with_output_mask_test_case_list = [(16, 8, 6, 6, x) for x in list(product([True, False], repeat=3))]


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_chlast_test_case_list)
@pytest.mark.skipif(is_lazy(), reason="https://jira.habana-labs.com/browse/SW-223808")
def test_hpu_chain_loop_conv_chlast_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel1_cpu = nn.Conv2d(C, K, R, stride, padding, 1, 1, bias)
    kernel1_copy = deepcopy(kernel1_cpu)
    kernel2_cpu = nn.Conv2d(K, K, R, stride, padding, 1, 1, bias)
    kernel2_copy = deepcopy(kernel2_cpu)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel1_hpu = kernel1_copy.to(hpu)
    kernel2_hpu = kernel2_copy.to(hpu)

    for _ in range(2):
        # cpu forward
        out_cpu_nchw_1 = kernel1_cpu(input_nchw)
        out_cpu_nchw_2 = kernel2_cpu(out_cpu_nchw_1)

        # hpu forward
        out_hpu_nhwc_1 = kernel1_hpu(input_c_last_hpu)
        out_hpu_nhwc_2 = kernel2_hpu(out_hpu_nhwc_1)

        # create bwd input tensor
        bwd_in = torch.randn(out_cpu_nchw_2.shape)
        out_cpu_bwd = out_cpu_nchw_1.grad_fn(out_cpu_nchw_2.grad_fn(bwd_in)[0])
        out_hpu_bwd = out_hpu_nhwc_1.grad_fn(
            out_hpu_nhwc_2.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last).to(hpu))[0]
        )
        np.testing.assert_allclose(
            out_hpu_bwd[0].view(out_cpu_bwd[0].shape).to(cpu).detach().numpy(),
            out_cpu_bwd[0].detach().numpy(),
            atol=0.01,
            rtol=0.01,
            equal_nan=True,
        )


@pytest.mark.parametrize("N, C, H, W, output_mask", conv_bwd_with_output_mask_test_case_list, ids=format_tc)
def test_hpu_conv_with_output_mask(N, C, H, W, output_mask):
    def check_grad(is_mask_enabled, grad, output_var_name):
        if is_mask_enabled:
            assert grad is not None, f"For a mask value equals to True, {output_var_name} cannot be equal to None"
        else:
            assert grad is None, f"For a mask value equals to False, {output_var_name} must be None"

    grad_output = torch.empty(size=[N, C, H, W], dtype=torch.float32).uniform_(-1, 1).to(hpu)
    input = torch.empty(size=[N, C, H, W], dtype=torch.float32).uniform_(-1, 1).to(hpu)
    weight = torch.empty(size=[C, C, H // 2, W // 2], dtype=torch.float32).uniform_(-1, 1).to(hpu)

    def conv_bwd(grad_output, input, weight, output_mask):
        # size of the bias (fourth input) must be correctly given, because PyTorch doesn't handle the size on its own what may lead to the errors on the backend level, more info: https://github.com/pytorch/pytorch/issues/119407
        return torch.ops.aten.convolution_backward(
            grad_output, input, weight, [C], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, output_mask
        )

    maybe_compiled_conv_bwd = compile_function_if_compile_mode(conv_bwd)
    grad_input, grad_weight, grad_bias = maybe_compiled_conv_bwd(grad_output, input, weight, output_mask)

    check_grad(output_mask[0], grad_input, "grad_input")
    check_grad(output_mask[1], grad_weight, "grad_weight")
    check_grad(output_mask[2], grad_bias, "grad_bias")
