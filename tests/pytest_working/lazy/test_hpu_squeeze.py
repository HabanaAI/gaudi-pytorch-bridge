# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import pytest
import torch
from fp8_utils import FP8_NAMES, check_native_fp8, dtype_from_string
from test_utils import compare_tensors, is_gaudi1

shapes = [(3, 1, 7, 4, 1), (1, 5, 1, 1, 8)]
dims = [(0, 3), (-1, 2), (1, -2, 0)]
dtypes = [torch.float, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes = dtypes + FP8_NAMES


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dims", dims)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("modify_view", [True, False])
def test_hpu_squeeze(shape, dims, dtype, modify_view):
    check_native_fp8(dtype)
    is_fp8 = dtype in FP8_NAMES
    dtype = dtype_from_string(dtype)

    input = (torch.randn(shape) * 5.0).to(dtype)
    input_hpu = input.to("hpu")
    result = torch.squeeze(input, dims)
    result_hpu = torch.squeeze(input_hpu, dims)

    if is_fp8:
        result = result.float()
        input = input.float()
        result_hpu = result_hpu.float()
        input_hpu = input_hpu.float()

    if modify_view:
        result.add_(1)
        result_hpu.add_(1)

    tol = 0 if dtype == torch.int else 0.001

    compare_tensors(result_hpu, result, tol, tol)
    compare_tensors(input_hpu, input, tol, tol)  # this will ensure that the view base is also updated


@pytest.mark.parametrize("shape", [(1,), (4,)])
@pytest.mark.parametrize("dim", [0, (0,)])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("modify_view", [True, False])
def test_hpu_squeeze_dim0(shape, dim, dtype, modify_view):
    check_native_fp8(dtype)
    is_fp8 = dtype in FP8_NAMES
    dtype = dtype_from_string(dtype)

    input = (torch.randn(shape) * 5.0).to(dtype)
    input_hpu = input.to("hpu")
    result = torch.squeeze(input, dim)
    result_hpu = torch.squeeze(input_hpu, dim)

    if is_fp8:
        result = result.float()
        input = input.float()
        result_hpu = result_hpu.float()
        input_hpu = input_hpu.float()

    if modify_view:
        result.add_(1)
        result_hpu.add_(1)

    tol = 0 if dtype == torch.int else 0.001

    compare_tensors(result_hpu, result, tol, tol)
    compare_tensors(input_hpu, input, tol, tol)  # this will ensure that the view base is also updated
