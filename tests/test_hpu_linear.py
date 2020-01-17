import torch
import torch.nn as nn
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed

# Multiply matrices NxC * CxK = NxK
test_case_list = [
    # N, C, K
    pytest.param(9, 8, 1, marks=pytest.mark.xfail(reason="SW-8820")),
    (1, 1, 3),
    pytest.param(1, 2, 3, marks=pytest.mark.xfail(reason="SW-8560")),
    pytest.param(8, 2, 3, marks=pytest.mark.xfail(reason="SW-8560")),
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
