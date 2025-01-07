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
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
from test_utils import hpu, is_gaudi2

pytestmark = pytest.mark.skipif(not is_gaudi2(), reason="Only Gaudi2 supports masked_batch_gemm op")


@pytest.mark.skip
@pytest.mark.parametrize("shape_A, shape_B", [([2, 3, 2, 4], [2, 3, 4, 8])])
@pytest.mark.parametrize("transA", [False, True])
@pytest.mark.parametrize("transB", [False, True])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_masked_batch_gemm(shape_A, shape_B, transA, transB, dtype):
    torch.manual_seed(12345)
    shapeA = shape_A.copy()
    shapeB = shape_B.copy()
    if transA:
        shapeA[-2], shapeA[-1] = shapeA[-1], shapeA[-2]
    if transB:
        shapeB[-2], shapeB[-1] = shapeB[-1], shapeB[-2]

    A = torch.randn(shapeA, dtype=dtype) * 5
    B = torch.randn(shapeB, dtype=dtype) * 5
    mask_A_shape = [shapeA[0], 1] + shapeA[-2:]
    mask_B_shape = [shapeB[0], 1] + shapeB[-2:]
    mask_A = torch.randn(mask_A_shape, dtype=dtype)
    mask_B = torch.randn(mask_B_shape, dtype=dtype)

    result = torch.ops.hpu.masked_batch_gemm(A.to(hpu), B.to(hpu), mask_A.to(hpu), mask_B.to(hpu), transA, transB).cpu()

    At = A.transpose(-2, -1) if transA else A
    mask_At = mask_A.transpose(-2, -1) if transA else mask_A
    Bt = B.transpose(-2, -1) if transB else B
    mask_Bt = mask_B.transpose(-2, -1) if transB else mask_B

    result_ref = torch.matmul(At, Bt) + torch.matmul(mask_At, mask_Bt)

    tol = 1e-3 if dtype == torch.float else 1e-1

    assert torch.allclose(result, result_ref, tol, tol)
