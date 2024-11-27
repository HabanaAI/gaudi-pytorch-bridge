###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################


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
