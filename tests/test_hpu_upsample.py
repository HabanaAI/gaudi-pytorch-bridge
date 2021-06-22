import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import itertools
import copy
import numpy as np
from test_utils import (
    evaluate_fwd_kernel,
    evaluate_fwd_bwd_kernel,
    reset_seed,
    compare_tensors,
)

op_list = [
    # op
    (F.interpolate),
]

test_case_list2 = [
    # N, H, W, C, scale_factor
    (2, 20, 30, 3, 2.0),
    (5, 40, 30, 4, 3.0)
    ]
test_case_list3 = [
    # N, H, W, C, scale_h, scale_w
    (2, 3, 4, 1, 1.0, 1.0),
    (2, 4, 3, 5, 2.0, 3.0),
    (5, 40, 30, 2, 0.75, 0.5),
    (5, 40, 30, 2, 0.5, 0.25)
    ]

test_case_list4 = [
    # N, H, W, C, out_h, out_w
    (1, 3, 4, 1, 6, 8),
    (1, 6, 8, 1, 18, 16),
    (1, 6, 8, 1, 3, 4),
    (1, 18, 16, 1, 6, 8),
    (1, 2, 1, 1, 8, 4),
    (1, 8, 4, 1, 2, 1),
    ]

hpu = torch.device('hpu')
cpu = torch.device('cpu')

@pytest.mark.parametrize("N, H, W, C, scale_factor", test_case_list2)
def test_interpolate2(N, H, W, C, scale_factor):
    in_tensor = torch.randn(N, C, H, W)
    hpu_result = torch.nn.functional.interpolate(in_tensor.to(hpu), scale_factor = (scale_factor, scale_factor), mode="nearest")
    cpu_result = torch.nn.functional.interpolate(in_tensor, scale_factor = (scale_factor, scale_factor), mode="nearest")
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=0.001)

@pytest.mark.parametrize("N, H, W, C, scale_h, scale_w", test_case_list3)
def test_interpolate3(N, H, W, C, scale_h, scale_w):
    in_tensor = torch.randn(N, C, H, W)
    hpu_result = torch.nn.functional.interpolate(in_tensor.to(hpu), scale_factor = (scale_h, scale_w), mode="nearest")
    cpu_result = torch.nn.functional.interpolate(in_tensor, scale_factor = (scale_h, scale_w), mode="nearest")
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=0.001)

@pytest.mark.parametrize("N, H, W, C, scale_h, scale_w", test_case_list3)
def test_interpolate_chnlast(N, H, W, C, scale_h, scale_w):
    in_tensor = torch.randn(N, C, H, W)
    in_tensor = in_tensor.to(memory_format=torch.channels_last)
    hpu_result = torch.nn.functional.interpolate(in_tensor.to(hpu), scale_factor = (scale_h, scale_w), mode="nearest")
    cpu_result = torch.nn.functional.interpolate(in_tensor, scale_factor = (scale_h, scale_w), mode="nearest")
    compare_tensors(hpu_result, cpu_result, atol=0.001, rtol=0.001)

@pytest.mark.parametrize("N, H, W, C, out_h, out_w", test_case_list4)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_interpolate_nearest2d_fwd(N, H, W, C, out_h, out_w, kernel_op):
    kernel_params = {
        'input': torch.randn(N, C, H, W, requires_grad=True),
        'size' : (out_h, out_w),
        'mode' :"nearest",
    }
    evaluate_fwd_kernel(
        kernel=kernel_op,  kernel_params=kernel_params
    )

@pytest.mark.parametrize("N, H, W, C, out_h, out_w", test_case_list4)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_interpolate_nearest2d_fwd_bwd(N, H, W, C, out_h, out_w, kernel_op):
    kernel_params = {
        'input': torch.randn(N, C, H, W, requires_grad=True),
        'size' : (out_h, out_w),
        'mode' :"nearest",
    }
    bwd_tensors = [torch.ones(N, C, out_h, out_w,)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel_op, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params
    )

@pytest.mark.parametrize("N, H, W, C, out_h, out_w", test_case_list4)
@pytest.mark.parametrize("kernel_op", op_list)
def test_hpu_interpolate_nearest2d_fwd_bwd_chnlast(N, H, W, C, out_h, out_w, kernel_op):
    kernel_params = {
        'input': torch.randn(N, C, H, W, requires_grad=True).to(memory_format=torch.channels_last),
        'size' : (out_h, out_w),
        'mode' :"nearest",
    }
    bwd_tensors = [torch.ones(N, C, out_h, out_w,).to(memory_format=torch.channels_last)]
    evaluate_fwd_bwd_kernel(
        kernel=kernel_op, tensor_list_bwd=bwd_tensors, kernel_params_fwd=kernel_params
    )

if __name__ == "__main__":
    test_hpu_interpolate_nearest2d_fwd_bwd_chnlast(*test_case_list4[0], *op_list[0])

