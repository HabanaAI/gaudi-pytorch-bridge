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
