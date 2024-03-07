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
import habana_frameworks.torch.core as htcore

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht
import numpy as np
import pytest
import torch
from fp8_utils import FP8_MAX, FP8_NAMES, IS_NATIVE_FP8, dtype_from_string, simulateFp8Precision
from test_utils import compare_tensors, is_gaudi1

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
@pytest.mark.parametrize("out_dtype", FP8_NAMES)
def test_cast_to_fp8(shape, dtype, out_dtype):
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0] * 2, shape[1])

    scale = torch.tensor(1.3, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale.to(dtype), out_dtype)
    unscaled_input = scaled_input_low_precision * scale_inv.to(dtype)

    scale_hpu = scale.to(hpu)
    scale_inv_hpu = scale_inv.to(hpu)
    casted = torch.empty(
        input.shape,
        dtype=out_dtype,
        device=hpu,
    )
    amax = torch.tensor(0, dtype=torch.float).to("hpu")
    torch.ops.hpu.cast_to_fp8(input.to(hpu), scale_hpu, False, casted, amax)
    uncasted = torch.ops.hpu.cast_from_fp8(casted, scale_inv_hpu, dtype)

    assert torch.equal(uncasted.cpu(), unscaled_input)

    assert amax.cpu() == torch.max(input.abs())


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("scale", [0.75])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES)
def test_fp8_gelu_v2(shape, scale, dtype, stochastic, is_scale, is_amax, out_dtype):
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_val = scale if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    gelu = torch.nn.GELU(approximate="tanh")
    gelu_res = gelu(input)
    scaled_gelu_low_precision = simulateFp8Precision(gelu_res * scale, out_dtype)
    result_cpu = scaled_gelu_low_precision * scale_inv
    retain_cpu = torch.tanh(
        torch.sqrt(torch.tensor(2 / np.pi, dtype=dtype)) * (input + 0.044715 * torch.pow(input, 3))
    ).to(dtype)

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    gelu_scaled, retain, amax = torch.ops.hpu.fp8_gelu_v2(input.to(hpu), scale_hpu, stochastic, is_amax, out_dtype)
    gelu_unscaled = torch.ops.hpu.cast_from_fp8(gelu_scaled, scale_inv_hpu, dtype).cpu()

    if stochastic:
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=0.26, atol=0.01)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=rtol, atol=0.0)
    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
    assert torch.equal(retain.cpu(), retain_cpu)


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("retain", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES)
def test_fp8_bgrad_dgelu_optional(shape, dtype, retain, is_scale, is_amax, out_dtype):
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    full_shape = (shape[0] * 2, shape[1])
    input_pos = torch.rand(shape, dtype=dtype, requires_grad=True) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    input_hpu = input.to(hpu)
    grad = torch.rand(full_shape, dtype=dtype)
    grad_hpu = grad.to(hpu)

    scale_val = 0.5 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()

    retain_tensor = None
    if retain:
        retain_tensor = (
            torch.tanh(torch.sqrt(torch.tensor(2 / np.pi, dtype=dtype)) * (input + 0.044715 * torch.pow(input, 3)))
            .to(dtype)
            .to(hpu)
        )
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    casted, bgrad, amax = torch.ops.hpu.fp8_bgrad_dgelu(
        grad_hpu, input_hpu, scale_hpu, retain_tensor, False, is_amax, out_dtype
    )

    gelu = torch.nn.GELU(approximate="tanh")
    gelu_res = gelu(input)
    gelu_bwd = gelu_res.grad_fn(grad)
    reduced = torch.sum(gelu_bwd, 0)

    scaled_input_low_precision = simulateFp8Precision(gelu_bwd * scale, out_dtype)
    unscaled_input = scaled_input_low_precision * scale_inv

    uncasted = torch.ops.hpu.cast_from_fp8(casted, scale_inv_hpu, dtype).cpu()

    assert torch.allclose(bgrad.cpu(), reduced)
    rtol = 0.01 if dtype == torch.bfloat16 else 0.0
    assert torch.allclose(uncasted, unscaled_input, rtol=rtol, atol=0.1)
    if is_amax:
        assert amax.cpu() == torch.max(gelu_bwd.abs())


# TODO analyze why single elements of outputs differ for torch.bfloat16
@pytest.mark.parametrize("shape", [(64, 96)])
@pytest.mark.parametrize("scale", [0.75])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES)
def test_fp8_dropout(shape, scale, dtype, is_scale, is_amax, out_dtype):
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    ratio = 0.3
    dropout_scale = torch.tensor(1.0 / (1.0 - ratio), dtype=dtype)

    scale = torch.tensor(scale, dtype=torch.float) if is_scale else torch.tensor(1.0)
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv = scale.reciprocal()

    dropout_scaled, mask, amax = torch.ops.hpu.fp8_dropout(input.to(hpu), ratio, scale_hpu, False, is_amax, out_dtype)
    dropout_unscaled = torch.ops.hpu.cast_from_fp8(dropout_scaled, scale_inv.to(hpu), dtype).cpu()

    scaled_input_low_precision = simulateFp8Precision(
        input * scale.to(dtype) * dropout_scale, out_dtype
    ) * scale_inv.to(dtype)
    result_ref = torch.where(mask.cpu().to(torch.bool), scaled_input_low_precision, 0.0)

    scaled_input_high_prec = input * dropout_scale
    dropout_high_prec = torch.where(mask.cpu().to(torch.bool), scaled_input_high_prec, 0.0)

    ones = torch.count_nonzero(mask.cpu())
    ratio_res = 1.0 - ones / mask.numel()

    assert torch.allclose(result_ref, dropout_unscaled)
    if is_amax:
        assert amax.cpu() == torch.max(dropout_high_prec.abs())
    assert torch.isclose(ratio_res, torch.tensor(ratio), rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "shapeA, shapeB",
    [((2, 3, 4, 2), (2, 3, 4, 8)), ((24, 12), (24, 36))],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_tensor", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", FP8_NAMES)
def test_fp8_gemm(shapeA, shapeB, bias, out_tensor, accumulate, scaleA, scaleB, dtype, fp8_dtype):
    fp8_dtype = dtype_from_string(fp8_dtype)
    torch.manual_seed(12345)
    if accumulate and not out_tensor:
        pytest.skip("Accumulate not supported without out_tensor")

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

    out = torch.full(out_shape, 1000.0, dtype=dtype)
    out_hpu = out.to(hpu) if out_tensor else torch.empty(out_shape, dtype=dtype, device=A_hpu.device)

    A8, _ = torch.ops.hpu.cast_to_fp8_v2(A_hpu, scaleA_hpu, False, False, fp8_dtype)
    B8, _ = torch.ops.hpu.cast_to_fp8_v2(B_hpu, scaleB_hpu, False, False, fp8_dtype)

    maybe_result = torch.ops.hpu.fp8_gemm(
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
    if out_tensor:
        result = out_hpu.cpu()
    else:
        result = maybe_result.cpu()

    percentage_diff = torch.abs((((result - result_ref) / result_ref) * 100).to(torch.int))
    assert np.amax(percentage_diff.numpy()) <= 15
