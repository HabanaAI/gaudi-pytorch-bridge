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
