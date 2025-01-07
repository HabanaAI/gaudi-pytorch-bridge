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

from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
from test_utils import cpu, format_tc, hpu, print_tensors

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

dilation_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, dilation, bias
    # (64, 224, 224, 3, 7, 7, 64, 2, 3, 1, False),
    # (64, 224, 224, 3, 7, 7, 64, 2, 3, 2, False),
    # (64, 56, 56, 64, 3, 3, 64, 1, 1, 2, False),
    # (64, 56, 56, 128, 3, 3, 128, 2, 1, 2, False)
    (4, 28, 28, 3, 2, 2, 16, 1, 0, 2, True),
    (2, 3, 4, 5, 2, 2, 6, 1, 0, 2, True),
    (8, 28, 28, 3, 2, 2, 16, 1, 1, 2, False),
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

conv_test_case_list = conv_chlast_test_case_list + [
    # N, H, W, C, R, S, K, str, pad, bias
    (None, 3, 4, 5, 2, 2, 6, 1, 0, True),
    (2, 3, None, 5, 2, 2, 6, 1, 0, True),
    (None, 3, None, 5, 2, 2, 6, 1, 0, True),
]

conv3d_test_case_list = [
    # N, D, H, W, C, T, R, S, K, stride, padding, bias
    (1, 8, 28, 28, 20, 3, 3, 3, 20, 1, 1, True),
    (2, 4, 28, 28, 20, 1, 1, 1, 40, 2, 0, False),
    # UNet3D layer
    (1, 16, 32, 64, 32, 3, 3, 3, 15, 2, 1, True),
]

conv_transpose_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, out_pad, bias
    (8, 28, 28, 3, 2, 2, 16, 1, 1, 0, False),
    (9, 28, 28, 1, 5, 5, 20, 2, 0, 1, True),
    (12, 11, 11, 20, 5, 5, 30, 1, 0, 0, True),
]

conv_transpose3d_test_case_list = [
    # N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias
    # UNet3D layer
    (1, 4, 4, 4, 320, 2, 2, 2, 320, 2, 0, 1, True),
    (1, 8, 8, 8, 320, 2, 2, 2, 256, 2, 0, 0, False),
]

conv_bwd_with_output_mask_test_case_list = [
    # N, H, W, C, output_mask
    (16, 8, 6, 6, [True, True, True]),
    (16, 8, 6, 6, [True, False, False]),
    (16, 8, 6, 6, [False, False, False]),
]

data_type_list = [(torch.float, 0.001)]


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, out_pad, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose(N, H, W, C, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C, K, R, stride, padding, out_pad, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    tt = out_nchw_hpu.to(cpu)
    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias",
    conv_transpose3d_test_case_list,
)
def test_hpu_conv_transpose3d(N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        output_padding=out_pad,
        bias=bias,
    )

    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    tt = out_nchw_hpu.to(cpu)
    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, out_pad, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose_chlast(N, H, W, C, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C, K, R, stride, padding, out_pad, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    tt = out_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias",
    conv_transpose3d_test_case_list,
)
def test_hpu_conv_transpose3d_chlast(N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        output_padding=out_pad,
        bias=bias,
    )

    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)

    # hpu forward
    out_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    tt = out_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, out_pad, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose_fwd_bwd(N, H, W, C, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C, K, R, stride, padding, out_pad, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    out_hpu_bwd_1 = out_hpu_bwd[1]
    np.testing.assert_allclose(
        out_hpu_bwd_1.to(cpu).detach().numpy(),
        out_cpu_bwd[1].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    if out_cpu_bwd[2] is not None:
        np.testing.assert_allclose(
            out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(),
            atol=0.01,
            rtol=0.01,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias",
    conv_transpose3d_test_case_list,
)
def test_hpu_conv_transpose3d_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        output_padding=out_pad,
        bias=bias,
    )
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)

    # hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    out_hpu_bwd_1 = out_hpu_bwd[1]
    np.testing.assert_allclose(
        out_hpu_bwd_1.to(cpu).detach().numpy(),
        out_cpu_bwd[1].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    if out_cpu_bwd[2] is not None:
        np.testing.assert_allclose(
            out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(),
            atol=0.01,
            rtol=0.01,
            equal_nan=True,
        )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, out_pad, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose_chlast_fwd_bwd(N, H, W, C, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C, K, R, stride, padding, out_pad, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last).to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    out_hpu_bwd_1 = out_hpu_bwd[1]
    np.testing.assert_allclose(
        out_hpu_bwd_1.to(cpu).detach().numpy(),
        out_cpu_bwd[1].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    if out_cpu_bwd[2] is not None:
        np.testing.assert_allclose(
            out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(),
            atol=0.01,
            rtol=0.01,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias",
    conv_transpose3d_test_case_list,
)
def test_hpu_conv_transpose3d_chlast_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, out_pad, bias):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        output_padding=out_pad,
        bias=bias,
    )
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last_3d).to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    out_hpu_bwd_1 = out_hpu_bwd[1]
    np.testing.assert_allclose(
        out_hpu_bwd_1.to(cpu).detach().numpy(),
        out_cpu_bwd[1].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )
    if out_cpu_bwd[2] is not None:
        np.testing.assert_allclose(
            out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(),
            atol=0.01,
            rtol=0.01,
            equal_nan=True,
        )


class NativeConv:
    def __init__(self, N, W, kernel_cpu, stride, padding, dilation, groups):
        def make_list(v):
            return [v, v]

        self.N = N
        self.W = W
        self.stride = make_list(stride)
        self.padding = make_list(padding)
        self.dilation = make_list(dilation)
        self.output_padding = make_list(0)
        self.groups = groups

        self.weight = kernel_cpu.weight
        self.bias = kernel_cpu.bias

    def __call__(self, input):
        def addNW(NW, t, axis):
            if NW:
                return t
            else:
                return torch.unsqueeze(t, axis)

        def addN(t):
            return addNW(self.N, t, 0)

        def addW(t):
            return addNW(self.W, t, -2)

        def delNW(NW, t, axis):
            if NW:
                return t
            else:
                return torch.squeeze(t, axis)

        def delN(t):
            return delNW(self.N, t, 0)

        def delW(t):
            return delNW(self.W, t, -2)

        return delN(
            delW(
                torch.ops.aten.convolution_overrideable(
                    addN(addW(input)),
                    addW(self.weight),
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    False,
                    self.output_padding,
                    self.groups,
                )
            )
        )

    def to(self, device):
        self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list, ids=format_tc)
@pytest.mark.parametrize("native", [False, True])
def test_hpu_conv(N, H, W, C, R, S, K, stride, padding, bias, dtype, tol, native):
    input_shape = tuple([x for x in [N, C, H, W] if x])
    input_nchw = torch.randn(input_shape, dtype=torch.float, requires_grad=True)

    if Verbose:
        print_tensors(["input_nchw"], [input_nchw])

    kernel_cpu = (nn.Conv2d if W else nn.Conv1d)(C, K, R, stride, padding, 1, 1, bias)
    kernel_hpu = NativeConv(N, W, kernel_cpu, stride, padding, 1, 1) if native else deepcopy(kernel_cpu)

    # cpu forward
    out_cpu_nchw = kernel_cpu(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_hpu.to(hpu)
    # hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    # print(out_cpu_nchw_hpu.shape, out_cpu_nchw_hpu.stride(), out_cpu_nchw.shape, out_cpu_nchw.stride())
    # hpu result permute since in channels_last, the kernel output is also in channels_last
    # but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nchw_hpu.to(cpu)

    if Verbose:
        print_tensors(["out_cpu_nchw", "out_hpu_nchw"], [out_cpu_nchw, tt])

    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv3d(N, D, H, W, C, T, R, S, K, stride, padding, bias, dtype, tol):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias,
    )

    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)

    # hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    # hpu result permute since in channels_last, the kernel output is also in channels_last
    # but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nchw_hpu.to(cpu)
    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias, dtype, tol):
    input_shape = tuple([x for x in [N, C, H, W] if x])
    input_nchw = torch.randn(input_shape, dtype=torch.float, requires_grad=True)

    kernel_nchw = (nn.Conv2d if W else nn.Conv1d)(C, K, R, stride, padding, 1, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].view(out_cpu_bwd[0].shape).to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv3d_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, bias, dtype, tol):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias,
    )
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, dilation, bias", dilation_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv_fwd_bwd_dilation(N, H, W, C, R, S, K, stride, padding, dilation, bias, dtype, tol):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C, K, R, stride, padding, dilation, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].view(out_cpu_bwd[0].shape).to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_chlast_test_case_list)
def test_hpu_conv_chlast(N, H, W, C, R, S, K, stride, padding, bias):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C, K, R, stride, padding, 1, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    # hpu result permute since in channels_last, the kernel output is also in channels_last
    # but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
def test_hpu_conv3d_chlast(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias,
    )

    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    # hpu result permute since in channels_last, the kernel output is also in channels_last
    # but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(
        tt.detach().numpy(),
        out_cpu_nchw.detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_chlast_test_case_list)
def test_hpu_conv_chlast_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias):
    input_nchw = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C, K, R, stride, padding, 1, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last).to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


# @pytest.mark.skip(reason="Device critical error")
@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
def test_hpu_conv3d_chlast_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    input_nchw = torch.randn((N, C, D, H, W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(
        in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias,
    )
    kernel_copy = deepcopy(kernel_nchw)
    # cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    # hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    # create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last_3d).to(hpu))
    np.testing.assert_allclose(
        out_hpu_bwd[0].to(cpu).detach().numpy(),
        out_cpu_bwd[0].detach().numpy(),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
    )


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_chlast_test_case_list)
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


@pytest.mark.parametrize("N, H, W, C, output_mask", conv_bwd_with_output_mask_test_case_list)
def test_hpu_conv_with_output_mask(N, H, W, C, output_mask):
    if False in output_mask:
        pytest.xfail("SW-177687")

    def check_grad(is_mask_enabled, grad, output_var_name):
        if is_mask_enabled:
            assert grad is not None, f"For a mask value equals to True, {output_var_name} cannot be equal to None"
        else:
            assert grad is None, f"For a mask value equals to False, {output_var_name} must be None"

    grad_output = torch.empty(size=[N, H, W, C], dtype=torch.float32).uniform_(-1, 1).to(hpu)
    input = torch.empty(size=[N, H, W, C], dtype=torch.float32).uniform_(-1, 1).to(hpu)
    weight = torch.empty(size=[H, H, W // 2, C // 2], dtype=torch.float32).uniform_(-1, 1).to(hpu)

    grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
        grad_output, input, weight, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, output_mask
    )

    check_grad(output_mask[0], grad_input, "grad_input")
    check_grad(output_mask[1], grad_weight, "grad_weight")
    check_grad(output_mask[2], grad_bias, "grad_bias")


if __name__ == "__main__":
    test_hpu_conv_fwd_bwd(*resnet50_test_case_list[0])
