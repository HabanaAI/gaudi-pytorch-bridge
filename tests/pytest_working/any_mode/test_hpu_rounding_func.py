###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
from test_utils import format_tc, is_gaudi1

dtypes = [torch.float32, torch.bfloat16]
integer_dtypes = [torch.int, torch.int16, torch.uint8, torch.int8]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("op", [torch.floor, torch.ceil, torch.trunc])
@pytest.mark.parametrize("dtype", dtypes + integer_dtypes, ids=format_tc)
def test_hpu_rounding_func(shape, op, dtype):
    def fn(input):
        return op(input)

    if dtype in integer_dtypes:
        cpu_input = torch.randint(low=0, high=100, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = op(cpu_input)
    hpu_output = fn(hpu_input).cpu()

    assert torch.equal(cpu_output, hpu_output)
