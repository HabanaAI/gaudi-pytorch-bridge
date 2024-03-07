# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************


import pytest
import torch
from test_utils import compare_tensors, evaluate_fwd_kernel

kthvalue_params_list = [
    ((8, 2), 3, 0),
    ((4, 4, 4, 2, 2), 2, 4),
    ((1, 9, 5), 2, -1),
]

dtype_list = [torch.float, torch.bfloat16, torch.int]


@pytest.mark.parametrize("self_shape, k_value, axis", kthvalue_params_list)
@pytest.mark.parametrize("dtype", dtype_list)
@pytest.mark.parametrize("keepdim", [True, False])
def test_hpu_kthvalue(self_shape, k_value, axis, keepdim, dtype):
    original_tensor = torch.randn(self_shape).to(dtype)
    kernel_params = {
        "input": original_tensor,
        "k": k_value,
        "dim": axis,
        "keepdim": keepdim,
    }
    kernel = torch.kthvalue

    (hpu_values, hpu_indices), (cpu_values, _) = evaluate_fwd_kernel(
        kernel, kernel_params=kernel_params, check_results=False
    )

    compare_tensors(hpu_values, cpu_values, atol=0, rtol=0, assert_enable=True)

    if not keepdim:
        hpu_indices = torch.unsqueeze(hpu_indices, axis)
        cpu_values = torch.unsqueeze(cpu_values, axis)
    gathered_values = torch.gather(original_tensor.to("hpu"), axis, hpu_indices)

    compare_tensors(gathered_values, cpu_values, atol=0, rtol=0, assert_enable=True)
