import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, compare_tensors

# N - batch
# H - input height
# W - input width
# C - input channels
# R - filter height
# S - filter width
# str - stride
pool_test_case_list = [
    # N, H, W, C, R, S, str_H, str_W
    (2, 3, 4, 5, 2, 2, 1, 1),
    (8, 28, 28, 3, 2, 2, 1, 1),
]


@pytest.mark.parametrize("N, H, W, C, R, S, str_H, str_W", pool_test_case_list)
def test_hpu_pool(N, H, W, C, R, S, str_H, str_W):
    # TODO: extend that test to all features
    kernel = F.max_pool2d
    kernel_params = {
        'input': torch.randn(N, C, H, W),
        'kernel_size': [R, S],
        'stride': [str_H, str_W],
        'return_indices': True
    }

    # don't check resuluts because indices can have different values
    hpu_result, cpu_result = evaluate_fwd_kernel(
        kernel=kernel, kernel_params=kernel_params, check_results=False)
    compare_tensors(hpu_result[0], cpu_result[0], atol=0.001, rtol=1.e-3)


if __name__ == '__main__':
    test_hpu_pool(*pool_test_case_list[0])
