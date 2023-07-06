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


if __name__ == "__main__":
    test_hpu_bitwise_op(*test_case_list[0])
