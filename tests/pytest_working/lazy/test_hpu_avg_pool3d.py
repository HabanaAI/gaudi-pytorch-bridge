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

import math

import pytest
import torch
import torch.nn.functional as F
from test_utils import evaluate_fwd_kernel

avg_pool3d_test_case_list = [
    # (N, C, D, H, W), kernel_size, stride, padding
    ((3, 3, 8, 16, 16), (3, 2, 2), (2, 1, 2), 1),
    ((2, 5, 6, 16, 20), 4, None, (1, 2, 2)),
    ((1, 2, 4, 4, 4), (1, 1, 1), 2, 0),
]
data_type_list = [(torch.float, 0.001)]


@pytest.mark.parametrize("input_shape, kernel_size, stride, padding", avg_pool3d_test_case_list)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
@pytest.mark.parametrize("divisor_override", [None, 4, -3])
@pytest.mark.parametrize("dtype, tol", data_type_list)
@pytest.mark.parametrize("chlast", [False, True])
def test_hpu_avg_pool_3d(
    input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tol, chlast
):
    input = torch.randn(input_shape).to(dtype)
    if chlast:
        input = input.contiguous(memory_format=torch.channels_last_3d)
    kernel_params = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
    }

    kernel = F.avg_pool3d

    evaluate_fwd_kernel(kernel=kernel, kernel_params=kernel_params, atol=tol, rtol=tol)
