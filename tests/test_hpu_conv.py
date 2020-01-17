import torch
import torch.nn as nn
import numpy as np
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed

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

# @torch.jit.script
@pytest.mark.parametrize("N, H, W, C, R, S, K, stride", conv_test_case_list)
def test_hpu_conv(N, H, W, C, R, S, K, stride):
    # TODO: extend that test to all features
    kernel = nn.Conv2d(C, K, R, stride)
    kernel_params = {
        'input': torch.randn(N, C, H, W),
    }
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


if __name__ == '__main__':
    test_hpu_conv(*conv_test_case_list[1])
