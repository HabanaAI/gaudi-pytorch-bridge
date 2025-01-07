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
