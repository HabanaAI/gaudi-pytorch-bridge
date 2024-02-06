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
import torch
import pytest
import habana_frameworks.torch.core as htcore
from itertools import combinations

from test_utils import is_gaudi1

def set_precision(dtype):
    atol = 0
    rtol = 0
    if dtype == torch.float16:
        atol = 1.0e-4
        rtol = 1.0e-4
    return atol, rtol

supported_dtypes = [torch.bfloat16, torch.float, torch.int, torch.short]
if not is_gaudi1():
    supported_dtypes.append(torch.half)

def generate_inputs(ndim):
    shapes = []
    shapes.append(tuple([1] * ndim))
    shapes.append(tuple([ndim * 2] * ndim))
    shapes.append(tuple(range(1, ndim * 2 + 1, 2)))
    shapes.append(tuple(range(ndim * 2, 0, -2)))

    if ndim != 3:
        dim = [0, 1, ndim-1, -2]
    else:
        dim = [0, 1, -1]
    dims = list(combinations(dim, 2))
    offset = [0, 1, ndim, ndim+1]
    return [shapes, dims, offset]

input_5d = generate_inputs(5)
input_4d = generate_inputs(4)
input_3d = generate_inputs(3)

def diagonal_test_generic(shape, dims, offset, dtype):
    input = torch.rand(shape).to(dtype=dtype)
    input_hpu = input.to('hpu')

    atol, rtol = set_precision(dtype)

    dim1 = dims[0]
    dim2 = dims[1]

    def fn(input, off, d1, d2):
        x = torch.diagonal(input, dim1=d1, dim2=d2, offset=off)
        # diagonal is classified as view OP. It requires some consuming OP
        # (e.g. torch.mul) to work in non-leaf mode.
        return torch.mul(x, 1)

    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    expected = cpu_compiled_fn(input, offset, dim1, dim2)
    result_hpu = hpu_compiled_fn(input_hpu, offset, dim1, dim2)
    result = result_hpu.to('cpu')

    assert torch.allclose(expected, result, atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", input_5d[0])
@pytest.mark.parametrize("dims", input_5d[1])
@pytest.mark.parametrize("offset", input_5d[2])
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_diagonal_5d(shape, dims, offset, dtype):
    diagonal_test_generic(shape, dims, offset, dtype)

@pytest.mark.parametrize("shape", input_4d[0])
@pytest.mark.parametrize("dims", input_4d[1])
@pytest.mark.parametrize("offset", input_4d[2])
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_diagonal_4d(shape, dims, offset, dtype):
    diagonal_test_generic(shape, dims, offset, dtype)

@pytest.mark.parametrize("shape", input_3d[0])
@pytest.mark.parametrize("dims", input_3d[1])
@pytest.mark.parametrize("offset", input_3d[2])
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_diagonal_3d(shape, dims, offset, dtype):
    diagonal_test_generic(shape, dims, offset, dtype)