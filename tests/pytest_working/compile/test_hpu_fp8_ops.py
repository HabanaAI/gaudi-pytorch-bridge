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
from fp8_utils import (
    dtype_from_string,
    simulateFp8Precision,
    IS_NATIVE_FP8,
    FP8_MAX,
    FP8_NAMES,
)
from test_utils import is_gaudi1, compare_tensors
import habana_frameworks.torch.core as htcore

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht

ht.disable_dynamic_shape()

pytestmark = [
    pytest.mark.skipif(is_gaudi1(), reason="Gaudi1 doesn't support fp8"),
    pytest.mark.skipif(
        not IS_NATIVE_FP8,
        reason="Native fp8 types are not supported in pytorch package.",
    ),
]


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES)
def test_cast_to_fp8_v2(shape, dtype, stochastic, is_amax, is_scale, out_dtype):
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0] * 2, shape[1])

    scale_val = 1.3 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(
        input * scale.to(dtype), out_dtype
    )
    unscaled_input = scaled_input_low_precision * scale_inv.to(dtype)

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None

    def fn(input, scale, scale_inv, stochastic, is_amax, out_dtype, dtype):
        casted, amax = torch.ops.hpu.cast_to_fp8_v2(
            input, scale, stochastic, is_amax, out_dtype
        )
        uncasted = torch.ops.hpu.cast_from_fp8(casted, scale_inv, dtype)
        return casted, amax, uncasted

    fn = torch.compile(fn, backend="aot_hpu_training_backend")
    casted, amax, uncasted = fn(
        input.to(hpu), scale_hpu, scale_inv_hpu, stochastic, is_amax, out_dtype, dtype
    )

    if stochastic:
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0)
    else:
        assert torch.equal(uncasted.cpu(), unscaled_input)

    if is_amax:
        assert amax.cpu() == torch.max(input.abs())


@pytest.mark.parametrize(
    "shapeA, shapeB",
    [((2, 3, 4, 2), (2, 3, 4, 8)), ((24, 12), (24, 36))],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", FP8_NAMES)
def test_fp8_gemm_v2(
    shapeA, shapeB, bias, accumulate, scaleA, scaleB, dtype, fp8_dtype
):
    fp8_dtype = dtype_from_string(fp8_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
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

    variant = "143" if fp8_dtype == torch.float8_e4m3fn else "152"
    if scaleA:
        scaleA_hpu = (FP8_MAX[variant] / max_A).to(hpu)
        scaleAInv = torch.reciprocal(scaleA_hpu)

    if scaleB:
        scaleB_hpu = (FP8_MAX[variant] / max_B).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)

    rank = len(shapeA)
    out_shape = shapeA[0 : (rank - 2)] + (shapeA[-1],) + (shapeB[-1],)
    bias_tensor = torch.rand(out_shape, dtype=dtype) * 10 + 30.0
    bias_tensor_hpu = bias_tensor.to(hpu) if bias else None

    if accumulate:
        out = torch.full(out_shape, 1000.0, dtype=dtype)
        out_hpu = out.to(hpu)
    else:
        out_hpu = None

    def fn(
        A_hpu,
        scaleA,
        B_hpu,
        scaleB,
        out_hpu,
        dtype,
        scaleA_inv,
        scaleB_inv,
        bias_tensor,
        accumulate,
    ):
        A8, _ = torch.ops.hpu.cast_to_fp8_v2(A_hpu, scaleA, False, False, fp8_dtype)
        B8, _ = torch.ops.hpu.cast_to_fp8_v2(B_hpu, scaleB, False, False, fp8_dtype)
        result = torch.ops.hpu.fp8_gemm_v2(
            A8,
            True,
            B8,
            False,
            out_hpu,
            dtype,
            scaleA_inv,
            scaleB_inv,
            bias_tensor,
            accumulate,
        )
        return result

    fn = torch.compile(fn, backend="aot_hpu_training_backend")
    result = fn(
        A_hpu,
        scaleA_hpu,
        B_hpu,
        scaleB_hpu,
        out_hpu,
        dtype,
        scaleAInv,
        scaleBInv,
        bias_tensor_hpu,
        accumulate,
    )

    result_ref = torch.matmul(A.transpose(-2, -1), B)

    if bias:
        result_ref = result_ref + bias_tensor
    if accumulate:
        result_ref = result_ref + out
    result = result.cpu()

    percentage_diff = torch.abs(
        (((result - result_ref) / result_ref) * 100).to(torch.int)
    )
    assert np.amax(percentage_diff.numpy()) <= 15


@pytest.mark.parametrize("shape", [(8, 2, 2, 5)])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + FP8_NAMES)
def test_in_place_interleave(shape, dtype):
    dtype = dtype_from_string(dtype)
    torch.manual_seed(12345)
    input = torch.randn(shape).to(dtype)
    input_hpu = input.to("hpu")

    indices = []
    for i in range(int(shape[0] / 4)):
        indices += [i] * 4
    index = torch.tensor(indices)

    def fn(input):
        torch.ops.hpu.in_place_interleave_(input)
        return input

    fn = torch.compile(fn, backend="aot_hpu_training_backend")

    fn(input_hpu)

    output_ref = torch.index_select(input, 0, index)

    compare_tensors(input_hpu, output_ref, atol=0.0, rtol=0.0)
