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
import habana_frameworks.torch.core as htcore

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
from fp8_utils import FP8_MAX, fp8_dtypes, simulateFp8Precision
from test_utils import compare_tensors, hpu, is_gaudi1

ht.disable_dynamic_shape()

pytestmark = pytest.mark.skipif(is_gaudi1(), reason="Gaudi1 doesn't support fp8")

hpu = torch.device("hpu")


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize("shape", [(4, 8)])
@pytest.mark.parametrize("scale", [1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("out_dtype", fp8_dtypes)
def test_cast_to_fp8(shape, scale, dtype, stochastic, transposed, allocate_out, out_dtype):
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0] * 2, shape[1])

    scale = torch.tensor(scale, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale, out_dtype)
    unscaled_input = scaled_input_low_precision * scale_inv

    casted = torch.empty(
        full_shape,
        dtype=out_dtype,
        device="hpu",
    )
    amax = torch.empty((), dtype=torch.float).to(hpu)
    torch.ops.hpu.cast_to_fp8(input.to(hpu), scale.to(hpu), stochastic, casted, amax)
    uncasted = torch.ops.hpu.cast_from_fp8(casted, scale_inv.to(hpu), dtype)

    if stochastic:
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=rtol, atol=0.0)
    assert amax.cpu() == torch.max(input.abs())


@pytest.mark.parametrize(
    "shapeA, shapeB",
    [((2, 3, 4, 2), (2, 3, 4, 8)), ((64, 48), (64, 112))],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes)
def test_fp8_gemm(shapeA, shapeB, bias, accumulate, scaleA, scaleB, dtype, fp8_dtype):
    A = torch.rand(shapeA, dtype=dtype) * 10 + 30.0
    A_hpu = A.to(hpu)
    max_A = torch.max(torch.abs(A)).to(torch.float)

    B = torch.rand(shapeB, dtype=dtype) * 10 + 30.0
    B_hpu = B.to(hpu)
    max_B = torch.max(torch.abs(B)).to(torch.float)

    scaleA_hpu = None
    scaleB_hpu = None
    scaleAInv = None
    scaleBInv = None

    if scaleA:
        scaleA_hpu = (FP8_MAX[fp8_dtype] / max_A).to(hpu)
        scaleAInv = torch.reciprocal(scaleA_hpu)

    if scaleB:
        scaleB_hpu = (FP8_MAX[fp8_dtype] / max_B).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)

    rank = len(shapeA)
    out_shape = shapeA[0 : (rank - 2)] + (shapeA[-1],) + (shapeB[-1],)
    bias_tensor = torch.rand(out_shape, dtype=dtype) * 10 + 30.0
    bias_tensor_hpu = bias_tensor.to(hpu) if bias else None

    out = torch.full(out_shape, 1000.0, dtype=dtype)
    out_hpu = out.to(hpu)

    A8, _ = torch.ops.hpu.cast_to_fp8_v2(A_hpu, scaleA_hpu, False, False, fp8_dtype)
    B8, _ = torch.ops.hpu.cast_to_fp8_v2(B_hpu, scaleB_hpu, False, False, fp8_dtype)

    torch.ops.hpu.fp8_gemm(
        A8,
        True,
        B8,
        False,
        out_hpu,
        dtype,
        scaleAInv,
        scaleBInv,
        bias_tensor_hpu,
        accumulate,
        out_hpu,
    )
    result_ref = torch.matmul(A.transpose(-2, -1), B)

    if bias:
        result_ref = result_ref + bias_tensor
    if accumulate:
        result_ref = result_ref + out

    percentage_diff = torch.abs((((out_hpu.cpu() - result_ref) / result_ref) * 100).to(torch.int))
    assert np.amax(percentage_diff.numpy()) <= 15
