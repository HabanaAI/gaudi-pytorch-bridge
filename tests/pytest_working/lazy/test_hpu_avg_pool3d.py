###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
