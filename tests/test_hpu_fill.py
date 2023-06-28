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

import torch
import pytest
from test_utils import compare_tensors
from test_utils import hpu

test_case_list = [
    # N, C, fill_val,
    (2, 10, 2.2),
    (2, 10, 0.0),
]


# @pytest.mark.xfail(reason="SW-8819")
@pytest.mark.parametrize("N, C, fill_val", test_case_list)
def test_hpu_fill(N, C, fill_val):
    cpu_tensor = torch.randn(N, C)
    hpu_tensor = cpu_tensor.to(hpu)

    cpu_tensor.fill_(fill_val)
    hpu_tensor.fill_(fill_val)
    compare_tensors(hpu_tensor, cpu_tensor, atol=0.001, rtol=1.e-3)

@pytest.mark.parametrize("N, C, fill_val", test_case_list)
def test_hpu_masked_fill(N, C, fill_val):
    cpu_tensor = torch.randn(N, C)
    hpu_tensor = cpu_tensor.to(hpu)

    mask = torch.randn(C) < 0
    val = torch.tensor(fill_val)

    cpu_tensor.masked_fill_(mask,val)
    hpu_tensor.masked_fill_(mask.to(hpu),val.to(hpu))
    compare_tensors(hpu_tensor, cpu_tensor, atol=0, rtol=0)

    cpu_tensor = torch.randn(N, C)
    hpu_tensor = cpu_tensor.to(hpu)

    cpu_tensor.masked_fill_(mask,fill_val)
    hpu_tensor.masked_fill_(mask.to(hpu),fill_val)
    compare_tensors(hpu_tensor, cpu_tensor, atol=0, rtol=0)

if __name__ == '__main__':
    test_hpu_fill(*test_case_list[0])
    test_hpu_masked_fill(*test_case_list[0])
