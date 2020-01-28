import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, compare_tensors
from numpy import floor

# N - batch
# H - input height
# W - input width
# C - input channels
# R - filter height
# S - filter width
# str - stride
mnist_dimensions = [
    # N, H, W, C, R, S, str_H, str_W
    (2, 24, 24, 20, 2, 2, 2, 2),
    (2, 8, 8, 50, 2, 2, 2, 2),
]
pool_test_case_list = [
    # N, H, W, C, R, S, str_H, str_W
    (8, 27, 27, 3, 3, 3, 2, 2),
] + mnist_dimensions


def output_size(spatial_size, pad, dilation, kernel_size, stride):
    return int(floor((spatial_size + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1))


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


@pytest.mark.xfail(reason="SW-9230")
@pytest.mark.parametrize("N, H, W, C, R, S, str_H, str_W", pool_test_case_list)
def test_hpu_pool_fwd_bwd(N, H, W, C, R, S, str_H, str_W):
    # TODO: extend that test to all features
    kernel = F.max_pool2d
    kernel_params_fwd = {
        'input': torch.randn(N, C, H, W, requires_grad=True),
        'kernel_size': [R, S],
        'stride': [str_H, str_W],
        'return_indices': True
    }
    bwd_tensors = [torch.randn(N, C, output_size(H, 0, 1, R, str_H), output_size(W, 0, 1, S, str_W))]
    # don't check fwd resuluts because indices can have different values
    (hpu_result_fwd, _), (cpu_result_fwd, _) = evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                                                                       kernel_params_fwd=kernel_params_fwd, check_results_fwd=False)
    compare_tensors(hpu_result_fwd[0], cpu_result_fwd[0], atol=0.001, rtol=1.e-3)


if __name__ == '__main__':
    test_hpu_pool(*pool_test_case_list[0])
