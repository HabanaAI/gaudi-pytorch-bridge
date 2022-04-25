import torch
import torch.nn as nn
import numpy as np
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed
from test_hpu_pool import output_size
from copy import deepcopy
import habana_frameworks.torch.core as htcore

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
    (64, 28, 28, 1, 5, 5, 20, 1, 0, True),
    (64, 11, 11, 20, 5, 5, 50, 1, 0, True),
]

resnet50_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, bias
    (64, 224, 224, 3, 7, 7, 64, 2, 3, False),
    (64, 56, 56, 64, 3, 3, 64, 1, 1, False),
    (64, 56, 56, 128, 3, 3, 128, 2, 1, False)
]

dilation_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, dilation, bias
    #(64, 224, 224, 3, 7, 7, 64, 2, 3, 1, False),
    #(64, 224, 224, 3, 7, 7, 64, 2, 3, 2, False),
    #(64, 56, 56, 64, 3, 3, 64, 1, 1, 2, False),
    #(64, 56, 56, 128, 3, 3, 128, 2, 1, 2, False)
    (8, 28, 28, 3, 2, 2, 16, 1, 0, 2, True),
    (2, 3, 4, 5, 2, 2, 6, 1, 0, 2, True),
    (8, 28, 28, 3, 2, 2, 16, 1, 1, 2, False)
]

conv_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, bias
    (2, 3, 4, 5, 2, 2, 6, 1, 0, True),
    (8, 28, 28, 3, 2, 2, 16, 1, 0, True),
    (8, 28, 28, 3, 2, 2, 16, 1, 1, False)
] + mnist_test_case_list + resnet50_test_case_list

conv3d_test_case_list = [
    # N, D, H, W, C, T, R, S, K, stride, padding, bias
    (1, 8, 28, 28, 20, 3, 3, 3, 20, 1, 1, True),
    (2, 4, 28, 28, 20, 1, 1, 1, 40, 2, 0, False),
    # UNet3D layer
    (1, 128, 128, 128, 32, 3, 3, 3, 64, 2, 1, True)
]

conv_transpose_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, bias
    (8, 28, 28, 3, 2, 2, 16, 1, 1, False),
    (64, 28, 28, 1, 5, 5, 20, 1, 0, True),
    (64, 11, 11, 20, 5, 5, 50, 1, 0, True)
]

conv_transpose3d_test_case_list = [
    # N, D, H, W, C, T, R, S, K, stride, padding, bias
    # UNet3D layer
    (1, 4, 4, 4, 320, 2, 2, 2, 320, 2, 0, True),
    (1, 8, 8, 8, 320, 2, 2, 2, 256, 2, 0, False)
]

data_type_list = [
  (torch.float, 0.001)
]

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose(N, H, W, C, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C,K,R,stride,padding, 0, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    tt = out_nchw_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv_transpose3d_test_case_list)
def test_hpu_conv_transpose3d(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias)

    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for DHWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))
    #hpu forward
    out_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    tt = out_nchw_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose_chlast(N, H, W, C, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C,K,R,stride,padding, 0, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    tt = out_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv_transpose3d_test_case_list)
def test_hpu_conv_transpose3d_chlast(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias)

    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))
    #hpu forward
    out_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    tt = out_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C,K,R,stride,padding, 0, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    out_hpu_bwd_1 = out_hpu_bwd[1]
    if not htcore.is_enabled_synapse_layout_handling():
        out_hpu_bwd_1 = out_hpu_bwd[1].permute(3,2,0,1)
    np.testing.assert_allclose(out_hpu_bwd_1.to(cpu).detach().numpy(),
                out_cpu_bwd[1].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    if (out_cpu_bwd[2] != None):
      np.testing.assert_allclose(out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv_transpose3d_test_case_list)
def test_hpu_conv_transpose3d_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for DHWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))
    #hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    out_hpu_bwd_1 = out_hpu_bwd[1]
    if not htcore.is_enabled_synapse_layout_handling():
        out_hpu_bwd_1= out_hpu_bwd[1].permute(4,3,0,1,2)
    np.testing.assert_allclose(out_hpu_bwd_1.to(cpu).detach().numpy(),
                out_cpu_bwd[1].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    if (out_cpu_bwd[2] != None):
      np.testing.assert_allclose(out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_transpose_test_case_list)
def test_hpu_conv_transpose_chlast_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose2d(C,K,R,stride,padding, 0, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last).to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    out_hpu_bwd_1 = out_hpu_bwd[1]
    if not htcore.is_enabled_synapse_layout_handling():
        out_hpu_bwd_1 = out_hpu_bwd[1].permute(3,2,0,1)
    np.testing.assert_allclose(out_hpu_bwd_1.to(cpu).detach().numpy(),
                out_cpu_bwd[1].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    if (out_cpu_bwd[2] != None):
      np.testing.assert_allclose(out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)


@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv_transpose3d_test_case_list)
def test_hpu_conv_transpose3d_chlast_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.ConvTranspose3d(in_channels=C,
        out_channels=K,
        kernel_size=(T, R, S),
        stride=stride,
        padding=padding,
        bias=bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWKC
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))
    #hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last_3d).to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    out_hpu_bwd_1 = out_hpu_bwd[1]
    if not htcore.is_enabled_synapse_layout_handling():
        out_hpu_bwd_1 = out_hpu_bwd[1].permute(4,3,0,1,2)
    np.testing.assert_allclose(out_hpu_bwd_1.to(cpu).detach().numpy(),
                out_cpu_bwd[1].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)
    if (out_cpu_bwd[2] != None):
      np.testing.assert_allclose(out_hpu_bwd[2].to(cpu).detach().numpy(),
            out_cpu_bwd[2].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv(N, H, W, C, R, S, K, stride, padding, bias, dtype, tol):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C,K,R,stride,padding, 1, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    #print(out_cpu_nchw_hpu.shape, out_cpu_nchw_hpu.stride(), out_cpu_nchw.shape, out_cpu_nchw.stride())
    #hpu result permute since in channels_last, the kernel output is also in channels_last
    #but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nchw_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv3d(N, D, H, W, C, T, R, S, K, stride, padding, bias, dtype, tol):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W), dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(in_channels=C,
        out_channels=K,
        kernel_size=(T,R,S),
        stride=stride,
        padding=padding,
        bias=bias)

    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for DHWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))

    #hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    #hpu result permute since in channels_last, the kernel output is also in channels_last
    #but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nchw_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias, dtype, tol):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C,K,R,stride,padding, 1, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].view(out_cpu_bwd[0].shape).to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv3d_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, bias, dtype, tol):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(in_channels=C,
        out_channels=K,
        kernel_size=(T,R,S),
        stride=stride,
        padding=padding,
        bias=bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for DHWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))
    #hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, dilation, bias", dilation_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_conv_fwd_bwd_dilation(N, H, W, C, R, S, K, stride, padding, dilation, bias, dtype, tol):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C,K,R,stride,padding, dilation, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_nchw_hpu = input_nchw.to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_cpu_nchw_hpu = kernel_nhwc_hpu(input_nchw_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nchw_hpu.grad_fn(bwd_in.to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].view(out_cpu_bwd[0].shape).to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
def test_hpu_conv_chlast(N, H, W, C, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C,K,R,stride,padding, 1, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    #hpu result permute since in channels_last, the kernel output is also in channels_last
    #but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
def test_hpu_conv3d_chlast(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(in_channels=C,
        out_channels=K,
        kernel_size=(T,R,S),
        stride=stride,
        padding=padding,
        bias=bias)

    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))
    #hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    #hpu result permute since in channels_last, the kernel output is also in channels_last
    #but for C=1, contiguous(memory_format=torch.channels_last) doesn't convert to channels_last
    tt = out_cpu_nhwc_hpu.to(cpu)
    np.testing.assert_allclose(tt.detach().numpy(), out_cpu_nchw.detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
def test_hpu_conv_chlast_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv2d(C,K,R,stride,padding, 1, 1, bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 1, 0))
    #hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last).to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, D, H, W, C, T, R, S, K, stride, padding, bias", conv3d_test_case_list)
def test_hpu_conv3d_chlast_fwd_bwd(N, D, H, W, C, T, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,D,H,W),dtype=torch.float, requires_grad=True)

    kernel_nchw = nn.Conv3d(in_channels=C,
        out_channels=K,
        kernel_size=(T,R,S),
        stride=stride,
        padding=padding,
        bias=bias)
    kernel_copy = deepcopy(kernel_nchw)
    #cpu forward
    out_cpu_nchw = kernel_nchw(input_nchw)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last_3d).to(hpu)
    kernel_nhwc_hpu = kernel_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel_nhwc_hpu.weight.data = kernel_nhwc_hpu.weight.data.permute((2, 3, 4, 1, 0))
    #hpu forward
    out_cpu_nhwc_hpu = kernel_nhwc_hpu(input_c_last_hpu)
    #create bwd input tensor
    bwd_in = torch.randn(out_cpu_nchw.shape)
    out_cpu_bwd = out_cpu_nchw.grad_fn(bwd_in)
    out_hpu_bwd = out_cpu_nhwc_hpu.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last_3d).to(hpu))
    np.testing.assert_allclose(out_hpu_bwd[0].to(cpu).detach().numpy(),
                out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
def test_hpu_chain_loop_conv_chlast_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias):
    hpu = torch.device('hpu')
    cpu = torch.device('cpu')
    input_nchw = torch.randn((N,C,H,W),dtype=torch.float, requires_grad=True)

    kernel1_cpu = nn.Conv2d(C,K,R,stride,padding, 1, 1, bias)
    kernel1_copy = deepcopy(kernel1_cpu)
    kernel2_cpu = nn.Conv2d(K,K,R,stride,padding, 1, 1, bias)
    kernel2_copy = deepcopy(kernel2_cpu)

    input_c_last_hpu = input_nchw.contiguous(memory_format=torch.channels_last).to(hpu)
    kernel1_hpu = kernel1_copy.to(hpu)
    kernel2_hpu = kernel2_copy.to(hpu)
    #Keep HPU weights metadata like sizes and strides same as in CPU, but data permuted for HWCK
    if not htcore.is_enabled_synapse_layout_handling():
        kernel1_hpu.weight.data = kernel1_hpu.weight.data.permute((2, 3, 1, 0))
        kernel2_hpu.weight.data = kernel2_hpu.weight.data.permute((2, 3, 1, 0))
    #weights_inter_hwck1 = kernel1_cpu.weight.data.to(hpu).permute((2, 3, 1, 0))
    #kernel1_hpu.weight.data = weights_inter_hwck1
    #weights_inter_hwck2 = kernel2_cpu.weight.data.to(hpu).permute((2, 3, 1, 0))
    #kernel2_hpu.weight.data = weights_inter_hwck2
    ##kernel1_cpu.weight.data = kernel1_hpu.weight.data.to(hpu).permute((2, 3, 1, 0))
    #kernel1_hpu.weight.data.copy_(weights_inter_hwck_1)
    #kernel2_cpu.weight.data = kernel1_hpu.weight.data.to(hpu).permute((2, 3, 1, 0))
    #kernel2_hpu.weight.data.copy_(weights_inter_hwck_2)

    for i in range(2):
        #cpu forward
        out_cpu_nchw_1 = kernel1_cpu(input_nchw)
        out_cpu_nchw_2 = kernel2_cpu(out_cpu_nchw_1)

        #hpu forward
        out_hpu_nhwc_1 = kernel1_hpu(input_c_last_hpu)
        out_hpu_nhwc_2 = kernel2_hpu(out_hpu_nhwc_1)

        #create bwd input tensor
        bwd_in = torch.randn(out_cpu_nchw_2.shape)
        out_cpu_bwd = out_cpu_nchw_1.grad_fn(out_cpu_nchw_2.grad_fn(bwd_in)[0])
        out_hpu_bwd = out_hpu_nhwc_1.grad_fn(out_hpu_nhwc_2.grad_fn(bwd_in.contiguous(memory_format=torch.channels_last).to(hpu))[0])
        np.testing.assert_allclose(out_hpu_bwd[0].view(out_cpu_bwd[0].shape).to(cpu).detach().numpy(),
                    out_cpu_bwd[0].detach().numpy(), atol=0.01, rtol=0.01, equal_nan=True)

if __name__ == '__main__':
    test_hpu_conv_fwd_bwd(*resnet50_test_case_list[0])
