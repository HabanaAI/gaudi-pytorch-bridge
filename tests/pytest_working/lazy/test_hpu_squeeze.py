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
from fp8_utils import fp8_dtypes
from test_utils import compare_tensors, is_gaudi1

shapes = [(3, 1, 7, 4, 1), (1, 5, 1, 1, 8)]
dims = [(0, 3), (-1, 2), (1, -2, 0)]
dtypes = [torch.float, torch.bfloat16, torch.int]
if not is_gaudi1():
    dtypes = dtypes + fp8_dtypes


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dims", dims)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("modify_view", [True, False])
def test_hpu_squeeze(shape, dims, dtype, modify_view):
    input = (torch.randn(shape) * 5.0).to(dtype)
    input_hpu = input.to("hpu")
    result = torch.squeeze(input, dims)
    result_hpu = torch.squeeze(input_hpu, dims)

    if dtype in fp8_dtypes:
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
    input = (torch.randn(shape) * 5.0).to(dtype)
    input_hpu = input.to("hpu")
    result = torch.squeeze(input, dim)
    result_hpu = torch.squeeze(input_hpu, dim)

    if dtype in fp8_dtypes:
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
