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
conv_test_case_list = [
    # N, H, W, C, R, S, K, str
    (2, 3, 4, 5, 2, 2, 6, 1),
    (8, 28, 28, 3, 2, 2, 16, 1),
]


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride", conv_test_case_list)
def test_hpu_conv(N, H, W, C, R, S, K, stride):
    # TODO: extend that test to all features
    kernel = nn.Conv2d(C, K, R, stride)
    kernel_params = {
        'input': torch.randn(N, C, H, W),
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, H, W, C, R, S, K, stride", conv_test_case_list)
def test_hpu_conv_fwd_bwd(N, H, W, C, R, S, K, stride):
    # TODO: extend that test to all features
    kernel = nn.Conv2d(C, K, R, stride)
    kernel_params_fwd = {
        'input': torch.randn(N, C, H, W, requires_grad=True)
    }
    bwd_tensors = [torch.randn(N, K, output_size(H, 0, 1, R, stride), output_size(W, 0, 1, S, stride))]
    (_, hpu_result_bwd), (_, cpu_result_bwd) = evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                                                                       kernel_params_fwd=kernel_params_fwd)


if __name__ == '__main__':
    test_hpu_conv(*conv_test_case_list[1])
