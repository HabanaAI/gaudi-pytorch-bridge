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
import torch
import pytest
import numpy as np
from habana_frameworks.torch.hpex.kernels.Fp8Ops import cast_to_fp8_te, fp8_gemm, fp8_transpose

MASK_FLOAT32 = torch.tensor(2145386496, dtype=torch.int) # 0 11111111 11000000000000000000000b
MASK_BFLOAT16 = torch.tensor(32736, dtype=torch.short) # 0 11111111 1100000b
FP8_MAX = torch.tensor(57344*0.9, dtype=torch.float)

def simulateFp8Precision(input):
    dtype = input.dtype
    if dtype == torch.float:
        mask = MASK_FLOAT32
        int_type = torch.int
    else:
        mask = MASK_BFLOAT16
        int_type = torch.short
    signs = torch.where(input < 0.0, -1.0, 1.0).to(dtype)
    asInt = input.view(int_type)
    masked = torch.bitwise_and(asInt, mask)
    return masked.view(dtype)*signs

@pytest.mark.parametrize("shape", [(64, 64, 768), (3, 4, 5)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_cast_to_fp8(shape, scale, dtype):
    hpu = torch.device("hpu")
    input = torch.randn(shape, dtype=dtype)
    input_fp8 = simulateFp8Precision(input)
    scale = torch.tensor(scale, dtype=torch.float)
    amax = torch.empty(1, dtype=torch.float).to(hpu)

    result = cast_to_fp8_te(input.to(hpu), scale.to(hpu), amax, True).cpu()
    result_ref = input_fp8*scale.to(dtype)

    assert amax.cpu() == torch.max(input.abs())

@pytest.mark.parametrize("shapeA, shapeB", [((2, 3, 4, 2), (2, 3, 4, 8)),
                                            ((5, 10, 6), (5, 10, 18)),
                                            ((64, 48), (64, 112))])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_tensor", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_gemm(shapeA, shapeB, bias, out_tensor, accumulate, dtype):
    if accumulate and not out_tensor:
        pytest.skip("Accumulate not supported without out_tensor")

    hpu = torch.device("hpu")
    A = torch.rand(shapeA, dtype=torch.float)*10 + 30.0
    A_hpu = A.to(hpu)
    max_A = torch.max(torch.abs(A))

    B = torch.rand(shapeB, dtype=torch.float)*10 + 30.0
    B_hpu = B.to(hpu)
    max_B = torch.max(torch.abs(B))

    scaleA_hpu = (FP8_MAX / max_A).to(hpu)
    scaleB_hpu = (FP8_MAX / max_B).to(hpu)

    scaleAInv = torch.reciprocal(scaleA_hpu)
    scaleBInv = torch.reciprocal(scaleB_hpu)

    amax_A = torch.empty((1,), dtype=torch.float).to(hpu)
    amax_B = torch.empty((1,), dtype=torch.float).to(hpu)

    rank = len(shapeA)
    out_shape = shapeA[0:(rank-2)] + (shapeA[-1],) + (shapeB[-1],)
    bias_tensor = torch.rand(out_shape, dtype=dtype)*10 + 30.0
    bias_tensor_hpu = bias_tensor.to(hpu) if bias else None

    out = torch.full(out_shape, 1000.0, dtype=dtype)
    out_hpu = out.to(hpu) if out_tensor else None

    A8 = cast_to_fp8_te(A_hpu, scaleA_hpu, amax_A, False)
    B8 = cast_to_fp8_te(B_hpu, scaleB_hpu, amax_B, False)

    maybe_result = fp8_gemm(A8, scaleAInv, B8, scaleBInv, out_dtype=dtype, out=out_hpu, bias=bias_tensor_hpu, use_bias=bias, accumulate=accumulate)
    result_ref = torch.matmul(A.transpose(-2, -1), B)

    if bias:
        result_ref = result_ref + bias_tensor
    if accumulate:
        result_ref = result_ref + out
    if out_tensor:
        result = out_hpu.cpu()
    else:
        result = maybe_result.cpu()

    percentage_diff = torch.abs((((result - result_ref) / result_ref)*100).to(torch.int))
    assert np.amax(percentage_diff.numpy()) <= 15

@pytest.mark.parametrize("shape", [(2, 4), (768, 1024)])
@pytest.mark.parametrize("is_out", [True, False])
def test_transpose(shape, is_out):
    hpu = torch.device("hpu")
    input = (torch.rand(shape)*50).to(torch.int8)
    input_hpu = input.to(hpu)
    if is_out:
        out = torch.empty(shape[1], shape[0], dtype=torch.int8).to(hpu)
        fp8_transpose(input_hpu, out)
    else:
        out = fp8_transpose(input_hpu)
    out_ref = input.t()

    assert np.array_equal(out.cpu(), out_ref)
