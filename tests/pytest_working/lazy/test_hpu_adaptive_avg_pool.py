###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import pytest
import torch
import torch.nn.functional as F
from test_utils import compare_tensors, evaluate_fwd_bwd_kernel, evaluate_fwd_kernel

# N - batch
# H - input height
# W - input width
# C - input channels
# Ho - Output height
# Wo - Output width
mnist_test_case_list = [
    # N, H, W, C, Ho, Wo
    (64, 24, 24, 20, 1, 1),
    (64, 7, 7, 50, 1, 1),
]

resnet50_test_case_list = [
    # N, H, W, C, Ho, Wo
    pytest.param(64, 112, 112, 64, 1, 1, marks=[pytest.mark.skip(reason="Crashes simulator")]),
    (8, 7, 7, 2048, 1, 1),
    (8, 7, 7, 2048, 1, 1),
]

adaptive_pool_test_case_list = (
    [
        # N, H, W, C, Ho, Wo
        (8, 27, 27, 3, 1, 1),
        (64, 15, 10, 8, 5, 4),
        (8, 7, 7, 20, 5, 5),
    ]
    + mnist_test_case_list
    + resnet50_test_case_list
)

data_type_list = [(torch.float, 0.001)]


@pytest.mark.parametrize("N, H, W, C, Ho, Wo", adaptive_pool_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
@pytest.mark.parametrize("use_batch", [True, False])
def test_hpu_adaptive_avgpool(N, H, W, C, Ho, Wo, dtype, tol, use_batch):
    shape = (N, C, H, W) if use_batch else (C, H, W)
    kernel_params = {
        "input": torch.randn(shape).to(dtype),
        "output_size": [Ho, Wo],
    }

    kernel = F.adaptive_avg_pool2d

    # don't check resuluts because indices can have different values
    hpu_result, cpu_result = evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params, check_results=False)
    compare_tensors(hpu_result[0], cpu_result[0], atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C, Ho, Wo", adaptive_pool_test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
@pytest.mark.parametrize("use_batch", [True, False])
def test_hpu_pool_fwd_bwd(N, H, W, C, Ho, Wo, dtype, tol, use_batch):
    shape = (N, C, H, W) if use_batch else (C, H, W)
    kernel_params_fwd = {
        "input": torch.randn(shape, requires_grad=True).to(dtype),
        "output_size": [Ho, Wo],
    }

    kernel = F.adaptive_avg_pool2d

    shape = (N, C, Ho, Wo) if use_batch else (C, Ho, Wo)
    bwd_tensors = [torch.randn(shape).to(dtype)]
    # don't check fwd results because indices can have different values
    (hpu_result_fwd, _), (cpu_result_fwd, _) = evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        check_results_fwd=False,
    )
    compare_tensors(hpu_result_fwd[0], cpu_result_fwd[0], atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C, Ho, Wo", adaptive_pool_test_case_list)
def test_hpu_chlast_pool(N, H, W, C, Ho, Wo):
    in_tensor = torch.randn(N, C, H, W)
    kernel_params = {
        "input": in_tensor.contiguous(memory_format=torch.channels_last),
        "output_size": [Ho, Wo],
    }

    kernel = F.adaptive_avg_pool2d

    # don't check results because indices can have different values
    hpu_result, cpu_result = evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params, check_results=False)
    compare_tensors(hpu_result[0], cpu_result[0], atol=0.001, rtol=1.0e-3)


@pytest.mark.parametrize("N, H, W, C, Ho, Wo", adaptive_pool_test_case_list)
def test_hpu_pool_chlast_fwd_bwd(N, H, W, C, Ho, Wo):
    # Pytorch (type='cpu'), kernel = <AsStridedBackward gives the following error for outputs size 1, 1
    # The size of tensor a (50) must match the size of tensor b (7) at non-singleton dimension
    if Ho == 1 and Wo == 1:
        return
    in_tensor = torch.randn(N, C, H, W, requires_grad=True)

    kernel_params_fwd = {
        "input": in_tensor.contiguous(memory_format=torch.channels_last),
        "output_size": [Ho, Wo],
    }

    kernel = F.adaptive_avg_pool2d

    bwd_tensor = torch.randn(N, C, H, W)
    bwd_tensors = [bwd_tensor.contiguous(memory_format=torch.channels_last)]
    # don't check fwd results because indices can have different values
    (hpu_result_fwd, hpu_result_bwd), (
        cpu_result_fwd,
        cpu_result_bwd,
    ) = evaluate_fwd_bwd_kernel(
        kernel=kernel,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        check_results_fwd=False,
    )
    compare_tensors(hpu_result_fwd[0], cpu_result_fwd[0], atol=0.001, rtol=1.0e-3)
    compare_tensors(hpu_result_bwd[0], cpu_result_bwd[0], atol=0.001, rtol=1.0e-3)


if __name__ == "__main__":
    test_hpu_adaptive_avgpool(*(adaptive_pool_test_case_list[0] + data_type_list[0]))
    test_hpu_pool_fwd_bwd(*(adaptive_pool_test_case_list[0] + data_type_list[0]))
    test_hpu_chlast_pool(*(adaptive_pool_test_case_list[0]))
    test_hpu_pool_chlast_fwd_bwd(*(adaptive_pool_test_case_list[0]))
