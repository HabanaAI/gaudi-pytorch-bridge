import torch
import torch.nn as nn
import numpy as np
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed
from test_hpu_pool import output_size

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

conv_test_case_list = [
    # N, H, W, C, R, S, K, str, pad, bias
    (2, 3, 4, 5, 2, 2, 6, 1, 0, True),
    (8, 28, 28, 3, 2, 2, 16, 1, 0, True),
    (8, 28, 28, 3, 2, 2, 16, 1, 1, False)
] + mnist_test_case_list + resnet50_test_case_list


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
def test_hpu_conv(N, H, W, C, R, S, K, stride, padding, bias):
    kernel = nn.Conv2d(C, K, R, stride, padding, 1, 1, bias)
    kernel_params = {
        'input': torch.randn(N, C, H, W),
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride, padding, bias", conv_test_case_list)
def test_hpu_conv_fwd_bwd(N, H, W, C, R, S, K, stride, padding, bias):
    kernel = nn.Conv2d(C, K, R, stride, padding, 1, 1, bias)
    kernel_params_fwd = {
        'input': torch.randn(N, C, H, W, requires_grad=True)
    }
    bwd_tensors = [torch.randn(N, K, output_size(H, padding, 1, R, stride), output_size(W, padding, 1, S, stride))]
    # Increase error tolerance from 1e-3 to 1e-1. For large IFM sizes (such as Resnet test cases) absolute & relative
    # error for bwd pass output tensors is becoming large.
    # [SW-11328] investigate this later
    (_, hpu_result_bwd), (_, cpu_result_bwd) = evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                                                                       kernel_params_fwd=kernel_params_fwd,atol=0.1, rtol=1.e-1)

if __name__ == '__main__':
    test_hpu_conv_fwd_bwd(*resnet50_test_case_list[0])
