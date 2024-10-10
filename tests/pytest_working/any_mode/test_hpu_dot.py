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

import pytest
import torch
from test_utils import compare_tensors, format_tc, hpu

tols = {torch.float: 1e-4, torch.bfloat16: 1e-2, torch.int: 0}


@pytest.mark.parametrize(
    "shape",
    [
        (5,),
        (128,),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int], ids=format_tc)
def test_index(shape, dtype):
    def wrapper_fn(input, other):
        return torch.dot(input, other)

    if pytest.mode == "compile":
        f_hpu = torch.compile(wrapper_fn, backend="hpu_backend")
    else:
        f_hpu = wrapper_fn

    if dtype == torch.int:
        input_tensor = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        other_tensor = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
    else:
        input_tensor = torch.rand(shape, dtype=dtype)
        other_tensor = torch.rand(shape, dtype=dtype)

    input_tensor_h = input_tensor.to(hpu)
    other_tensor_h = other_tensor.to(hpu)

    result_c = wrapper_fn(input=input_tensor, other=other_tensor)
    result_h = f_hpu(input=input_tensor_h, other=other_tensor_h)

    compare_tensors(result_h, result_c, tols[dtype], tols[dtype])
