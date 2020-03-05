import torch
import torch.nn as nn
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed

mnist_test_cast_list = [
    # N, C, K
    (64, 450, 500),
    (64, 500, 10),
]

# Multiply matrices NxC * CxK = NxK
test_case_list = [
    # N, C, K
    # (10, 20, 30),
    (800, 500, 10),
] + mnist_test_cast_list


@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=True)
    kernel_params = {'input': torch.randn(N, C)}
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear_fwd_bwd(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=True)
    kernel_params_fwd = {'input': torch.randn(N, C)}
    bwd_tensors = [torch.randn(N, K)]
    evaluate_fwd_bwd_kernel(kernel=kernel, kernel_params_fwd=kernel_params_fwd, tensor_list_bwd=bwd_tensors)


@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear_no_bias(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=False)
    kernel_params = {'input': torch.randn(N, C)}
    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params)


@pytest.mark.parametrize("N, C, K", test_case_list)
def test_hpu_linear_no_bias_fwd_bwd(N, C, K):
    kernel = nn.Linear(in_features=C, out_features=K, bias=False)
    kernel_params_fwd = {'input': torch.randn(N, C)}
    bwd_tensors = [torch.randn(N, K)]
    evaluate_fwd_bwd_kernel(kernel=kernel, kernel_params_fwd=kernel_params_fwd, tensor_list_bwd=bwd_tensors)


if __name__ == '__main__':
    test_hpu_linear_no_bias(*test_case_list[-1])
