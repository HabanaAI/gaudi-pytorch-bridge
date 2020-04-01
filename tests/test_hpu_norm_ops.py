import torch
import torch.nn.functional as F
import pytest
from test_utils import evaluate_fwd_kernel, evaluate_fwd_bwd_kernel, reset_seed, evaluate_fwd_inplace_kernel

# N - batch
# H - input height
# W - input width
# C - input channels
batch_norm_test_case_list_2d = [
    # N, H, W, C
    (16, 224, 224, 3),
]

batch_norm_test_case_list_1d = [
    # N, C
    (16, 128),
]

batch_norm_test_case_list_1d_ncl = [
    # N, C, L
    (32, 64, 5),
]

# Note: TODO SW-11483 # copy_kernel set to False, as pushing the kernel to the device throws due to dependencies
# on resize. Further copying the kernel to device is not needed for this eager mode testing
@pytest.mark.parametrize("N, H, W, C", batch_norm_test_case_list_2d)
def test_hpu_batch_norm_2d_fwd_bwd(N, H, W, C):
    kernel = torch.nn.BatchNorm2d(C)
    kernel_params_fwd = {'input': torch.randn(N, C, H, W, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C, H, W)]

    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd, copy_kernel=False)


@pytest.mark.parametrize("N, C", batch_norm_test_case_list_1d)
def test_hpu_batch_norm_1d_fwd_bwd(N, C):
    kernel = torch.nn.BatchNorm1d(C)
    kernel_params_fwd = {'input': torch.randn(N, C, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C)]

    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd, copy_kernel=False)


@pytest.mark.parametrize("N, C, L", batch_norm_test_case_list_1d_ncl)
def test_hpu_batch_norm_1d_ncl_fwd_bwd(N, C, L):
    kernel = torch.nn.BatchNorm1d(C)
    kernel_params_fwd = {'input': torch.randn(N, C, L, requires_grad=True)}
    bwd_tensors = [torch.randn(N, C, L)]

    evaluate_fwd_bwd_kernel(kernel=kernel, tensor_list_bwd=bwd_tensors,
                            kernel_params_fwd=kernel_params_fwd, copy_kernel=False)


if __name__ == '__main__':
    test_hpu_batch_norm_2d_fwd_bwd(*batch_norm_test_case_list_2d[0])
    test_hpu_batch_norm_1d_fwd_bwd(*batch_norm_test_case_list_1d[0])
    test_hpu_batch_norm_1d_ncl_fwd_bwd(*batch_norm_test_case_list_1d_ncl[0])
