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

test_case_list = [
    # C, H, W
    (5, 3, 4, torch.bitwise_and),
    (5, 3, 1, torch.bitwise_and),
    (5, 1, 4, torch.bitwise_and),
    (1, 3, 4, torch.bitwise_and),
    (5, 3, 4, torch.bitwise_or),
    (5, 3, 1, torch.bitwise_or),
    (5, 1, 4, torch.bitwise_or),
    (1, 3, 4, torch.bitwise_or),
    (5, 3, 4, torch.bitwise_xor),
    (5, 3, 1, torch.bitwise_xor),
    (5, 1, 4, torch.bitwise_xor),
    (1, 3, 4, torch.bitwise_xor),
    (5, 3, 4, torch.bitwise_not),
]


@pytest.mark.parametrize("C, H, W, bitwise_op", test_case_list)
def test_hpu_bitwise_op(C, H, W, bitwise_op):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randint(-10, 10, (5, 3, 4)) > 0
    if bitwise_op != torch.bitwise_not:
        kernel_params_fwd["other"] = torch.randint(-10, 10, (C, H, W)) > 0
    evaluate_fwd_kernel(kernel=bitwise_op, kernel_params=kernel_params_fwd)
