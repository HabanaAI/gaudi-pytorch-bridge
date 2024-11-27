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
from test_utils import format_tc

dtypes = [torch.float, torch.bfloat16, torch.int8, torch.int32, torch.long]


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("p", [1, 5])
def test_addmm(dtype, n, m, p):
    input_shape = (n, p)
    mat1_shape = (n, m)
    mat2_shape = (m, p)

    def fn(input, mat1, mat2):
        return torch.addmm(input, mat1, mat2)

    compiled_fn_hpu = torch.compile(fn, backend="hpu_backend")

    if dtype.is_floating_point:
        input = torch.randn(input_shape, dtype=dtype)
        mat1 = torch.randn(mat1_shape, dtype=dtype)
        mat2 = torch.randn(mat2_shape, dtype=dtype)
    else:
        input = torch.randint(low=-128, high=127, size=input_shape, dtype=dtype)
        mat1 = torch.randint(low=-128, high=127, size=mat1_shape, dtype=dtype)
        mat2 = torch.randint(low=-128, high=127, size=mat2_shape, dtype=dtype)

    expected = fn(input.cpu(), mat1.cpu(), mat2.cpu())
    result = compiled_fn_hpu(input, mat1, mat2)
    assert torch.allclose(result.cpu(), expected)


@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("m", [5])
@pytest.mark.parametrize("p", [5])
def test_inplace_addmm_with_view_input(dtype, n, m, p):
    input_shape = (p, n)
    mat1_shape = (n, m)
    mat2_shape = (m, p)

    def fn(input, mat1, mat2):
        input = torch.permute(input, [1, 0])
        return input.addmm_(mat1, mat2)

    torch._dynamo.reset()
    compiled_fn_hpu = torch.compile(fn, backend="hpu_backend")

    if dtype.is_floating_point:
        input = torch.randn(input_shape, dtype=dtype)
        mat1 = torch.randn(mat1_shape, dtype=dtype)
        mat2 = torch.randn(mat2_shape, dtype=dtype)
    else:
        input = torch.randint(low=-128, high=127, size=input_shape, dtype=dtype)
        mat1 = torch.randint(low=-128, high=127, size=mat1_shape, dtype=dtype)
        mat2 = torch.randint(low=-128, high=127, size=mat2_shape, dtype=dtype)

    expected = fn(input.cpu(), mat1.cpu(), mat2.cpu())
    result = compiled_fn_hpu(input, mat1, mat2)
    assert torch.allclose(result.cpu(), expected)
