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
from test_utils import compare_tensors, is_gaudi1

dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes += [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("op", [torch.minimum, torch.maximum])
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_minimum_maximum(shape, op, dtype):
    def fn(input, other):
        return op(input, other)

    if dtype == torch.int:
        cpu_input = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        cpu_other = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
    else:
        cpu_input = torch.randn(shape).to(dtype)
        cpu_other = torch.randn(shape).to(dtype)

    hpu_input = cpu_input.to("hpu")
    hpu_other = cpu_other.to("hpu")

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        cpu_input = cpu_input.float()
        cpu_other = cpu_other.float()

    if pytest.mode == "compile":
        fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = op(cpu_input, cpu_other)
    hpu_output = fn(hpu_input, hpu_other).cpu()

    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        hpu_output = hpu_output.float()

    assert torch.equal(cpu_output, hpu_output)
