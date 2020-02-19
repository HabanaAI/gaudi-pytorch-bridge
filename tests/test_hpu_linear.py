import torch
import torch.nn as nn
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, compare_tensors

# Multiply matrices NxC * CxK = NxK
test_case_list = [
    # N, C, K
    (10, 20, 30),
    (800, 500, 10),
]


@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=True)
    kernel_params = {'input': torch.randn(N, C)}
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear_no_bias(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=False)
    kernel_params = {'input': torch.randn(N, C)}
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


if __name__ == '__main__':
    test_hpu_linear_no_bias(*test_case_list[-1])
