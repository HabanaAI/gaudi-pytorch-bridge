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

import random

import pytest
import torch
from test_utils import is_gaudi1

supported_dtypes = [torch.float, torch.bfloat16, torch.long, torch.int, torch.short]
if not is_gaudi1():
    supported_dtypes.append(torch.half)


def generate_inputs(shape, dtype):
    if dtype in [torch.float, torch.bfloat16, torch.half]:
        input = torch.randn(shape, dtype=dtype)
        other = random.uniform(-2.0, 2.0)
    else:
        input = torch.randint(0, 10, shape, dtype=dtype)
        other = random.randint(0, 10)

    return (input, other, input.to("hpu"))


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_hpu_add_scalar(dtype):
    input, other, input_hpu = generate_inputs((8, 12), dtype)

    def op(a, b):
        return torch.add(a, b)

    if pytest.mode == "compile":
        op = torch.compile(op, backend="hpu_backend")

    result = op(input, other)
    result_hpu = op(input_hpu, other)

    assert torch.allclose(result_hpu.cpu(), result)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_hpu_add_scalar_inplace(dtype):

    input, other, input_hpu = generate_inputs((8, 12), dtype)

    def op(a, b):
        return a.add_(b)

    if pytest.mode == "compile":
        op = torch.compile(op, backend="hpu_backend")

    op(input, other)
    op(input_hpu, other)

    assert torch.allclose(input_hpu.cpu(), input)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_hpu_add_scalar_out(dtype):

    input, other, input_hpu = generate_inputs((8, 12), dtype)
    out = torch.empty_like(input)
    out_hpu = torch.empty_like(input_hpu)

    def op(a, b, out):
        return torch.add(a, b, out=out)

    if pytest.mode == "compile":
        op = torch.compile(op, backend="hpu_backend")

    op(input, other, out)
    op(input_hpu, other, out_hpu)

    assert torch.allclose(out_hpu.cpu(), out)
