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
from test_utils import evaluate_fwd_kernel

shapes = [
    ((2, 4, 6), (2, 4, 6)),
    ((5, 7), (3, 5, 7)),
    ((2, 5, 8), (5, 8)),
    ((4, 1, 6), (4, 5, 1)),
    ((3, 1), (2, 1, 5)),
]


@pytest.mark.parametrize("input_shape, other_shape", shapes)
@pytest.mark.parametrize("op", [torch.bitwise_left_shift, torch.bitwise_right_shift])
@pytest.mark.parametrize("dtype", [torch.int, torch.int8, torch.uint8, torch.short])
def test_hpu_bitwise_shift_op(input_shape, other_shape, op, dtype):
    if dtype in [torch.int8, torch.uint8]:
        pytest.xfail(reason="SW-158652")

    torch.manual_seed(12345)
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randint(0, 10, input_shape, dtype=dtype)
    kernel_params_fwd["other"] = torch.randint(0, 10, other_shape, dtype=dtype)

    evaluate_fwd_kernel(kernel=op, kernel_params=kernel_params_fwd)
