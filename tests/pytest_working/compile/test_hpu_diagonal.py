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

from itertools import combinations

import habana_frameworks.torch.core as htcore
import pytest
import torch
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

#             shape                 dims        offset
input5D = [[(10, 10, 10, 10, 10), (0, 1), 0]]
input4D = [
    [(2, 2, 2, 2, 2), (1, 0), 5],
    [(5, 5, 5, 5), (-1, 2), 3],
    [(1, 2, 3, 4), (-1, -2), 2],
    [(1, 1, 1, 1), (3, 0), -1],
]
input3D = [[(5, 3, 1), (0, 2), 2], [(2, 2, 2), (0, 1), 0]]


def diagonal_test_generic(shape, dims, offset, dtype):
    input = torch.rand(shape).to(dtype=dtype)
    input_hpu = input.to("hpu")

    atol, rtol = set_precision(dtype)

    dim1 = dims[0]
    dim2 = dims[1]

    def fn(input, off, d1, d2):
        x = torch.diagonal(input, dim1=d1, dim2=d2, offset=off)
        # diagonal is classified as view OP. It requires some consuming OP
        # (e.g. torch.mul) to work in non-leaf mode.
        return torch.mul(x, 1)

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    expected = fn(input, offset, dim1, dim2)
    result_hpu = hpu_compiled_fn(input_hpu, offset, dim1, dim2)
    result = result_hpu.to("cpu")

    assert torch.allclose(expected, result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("input", input5D)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
def test_diagonal_5D(input, dtype):
    shape = input[0]
    dims = input[1]
    offset = input[2]

    diagonal_test_generic(shape, dims, offset, dtype)


@pytest.mark.parametrize("input", input4D)
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_diagonal_4D(input, dtype):
    shape = input[0]
    dims = input[1]
    offset = input[2]

    diagonal_test_generic(shape, dims, offset, dtype)


@pytest.mark.parametrize("input", input3D)
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_diagonal_3D(input, dtype):
    shape = input[0]
    dims = input[1]
    offset = input[2]

    diagonal_test_generic(shape, dims, offset, dtype)
