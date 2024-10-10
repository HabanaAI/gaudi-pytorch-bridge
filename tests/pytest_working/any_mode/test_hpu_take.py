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
from test_utils import format_tc

dtypes_inputs = [torch.float32, torch.bfloat16, torch.int32]
dtypes_indicies = [torch.int32, torch.long]


@pytest.mark.parametrize(
    "shape, repeats",
    [
        ([3], [2]),
        ([3, 2, 4], [6]),
        ([3, 2, 3, 4, 2], [9]),
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("dtypes_inputs", dtypes_inputs, ids=format_tc)
@pytest.mark.parametrize("dtypes_indicies", dtypes_indicies, ids=format_tc)
def test_hpu_take(shape, repeats, dtypes_inputs, dtypes_indicies):
    if pytest.mode == "compile" and dtypes_indicies == torch.int32:
        pytest.skip(reason="Indicies tensor of dtype int32 is supported only in eager and lazy mode on hpu")

    if dtypes_inputs == torch.int32:
        input_tensor = torch.randint(size=shape, low=-10, high=10, dtype=dtypes_inputs)
    else:
        input_tensor = torch.rand(shape, dtype=dtypes_inputs)
    # CPU torch.take only accepts indicies as a LongTensor
    indicies = torch.randint(0, input_tensor.numel() - 1, repeats, dtype=torch.long)

    input_tensor_h = input_tensor.to("hpu")
    indicies_h = indicies.to("hpu").to(dtypes_indicies)

    def fn(input_tensor, indicies):
        return torch.take(input_tensor, indicies)

    if pytest.mode == "compile":
        f_hpu = torch.compile(fn, backend="hpu_backend")
    else:
        f_hpu = fn

    result_c = fn(input_tensor=input_tensor, indicies=indicies)
    result_h = f_hpu(input_tensor=input_tensor_h, indicies=indicies_h)

    assert torch.equal(result_c, result_h.to("cpu"))
