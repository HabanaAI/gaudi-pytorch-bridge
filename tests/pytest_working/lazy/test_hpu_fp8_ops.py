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
from fp8_utils import (
    FP8_MAX,
    FP8_NAMES_LEGACY,
    check_native_fp8,
    dtype_from_string,
    simulateFp8Precision,
    variant_from_dtype,
)
from habana_frameworks.torch.hpex.kernels.Fp8Ops import (
    cast_from_fp8,
    cast_to_fp8,
    cast_to_fp8_v2,
    fp8_cast_transpose_bgrad_dgelu_fused,
    fp8_cast_transpose_bgrad_fused,
    fp8_cast_transpose_fused,
    fp8_gelu,
    fp8_gemm,
    fp8_transpose,
    layernorm_fwd_fp8,
)
from test_utils import compare_tensors, hpu, is_gaudi1

ht.disable_dynamic_shape()

pytestmark = pytest.mark.skipif(is_gaudi1(), reason="Gaudi1 doesn't support fp8")


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize("shape", [(4, 8)])
@pytest.mark.parametrize("scale", [1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("transposed, allocate_out", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_cast_to_fp8(shape, scale, dtype, stochastic, transposed, allocate_out, out_dtype):
    check_native_fp8(out_dtype)
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0] * 2, shape[1])

    scale = torch.tensor(scale, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale, out_dtype)
    unscaled_input = scaled_input_low_precision * scale_inv

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    if transposed:
        if allocate_out:
            empty_dtype = torch.int8 if out_dtype is None else out_dtype
            casted = torch.empty(
                full_shape,
                dtype=empty_dtype,
                device="hpu",
            )
            casted_t = torch.empty(
                (full_shape[1], full_shape[0]),
                dtype=empty_dtype,
                device="hpu",
            )
            fp8_cast_transpose_fused(
                input.to(hpu),
                scale.to(hpu),
                amax[1][2],
                stochastic,
                casted,
                casted_t,
                out_dtype,
            )
        else:
            casted, casted_t = fp8_cast_transpose_fused(
                input.to(hpu),
                scale.to(hpu),
                amax[1][2],
                stochastic,
                out_dtype=out_dtype,
            )
        assert torch.equal(casted.cpu().t().to(dtype), casted_t.cpu().to(dtype))
    else:
        casted = cast_to_fp8(input.to(hpu), scale.to(hpu), amax[1][2], stochastic, out_dtype)
    uncasted = cast_from_fp8(casted, scale_inv.to(hpu), dtype)

    if stochastic:
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=rtol, atol=0.0)
    assert amax.cpu()[1][2] == torch.max(input.abs())


@pytest.mark.parametrize("shape", [(72, 56)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_cast_to_fp8_transpose_optional(shape, dtype, is_scale, is_amax):
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_val = 0.5 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    amax = torch.zeros((2, 3), dtype=torch.float).to(hpu)
    amax_tensor = amax[1][2] if is_amax else None
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    casted, casted_t = fp8_cast_transpose_fused(input.to(hpu), scale_hpu, amax_tensor, False)
    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype)

    assert torch.equal(casted.cpu().t(), casted_t.cpu())
    rtol = 0.01 if dtype == torch.bfloat16 else 0.0
    assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=rtol, atol=0.0)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())


@pytest.mark.parametrize("shape", [(72, 56)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_cast_to_fp8_optional(shape, dtype, is_scale, is_amax):
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_val = 0.5 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    amax = torch.zeros((2, 3), dtype=torch.float).to(hpu)
    amax_tensor = amax[1][2] if is_amax else None
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    casted = cast_to_fp8(input.to(hpu), scale_hpu, amax_tensor, False)
    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype)

    rtol = 0.01 if dtype == torch.bfloat16 else 0.0
    assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=rtol, atol=0.0)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("scale", [0.75])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_fp8_cast_transpose_bgrad(shape, scale, dtype, stochastic, out_dtype):
    check_native_fp8(out_dtype)
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    input_hpu = input.to(hpu)

    scale = torch.tensor(scale, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale, out_dtype)
    unscaled_input = scaled_input_low_precision * scale_inv

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    bgrad, casted, casted_t = fp8_cast_transpose_bgrad_fused(
        input_hpu, scale.to(hpu), amax[1][2], stochastic, out_dtype
    )

    reduced = torch.sum(input_hpu, 0)
    uncasted = cast_from_fp8(casted, scale_inv.to(hpu), dtype)

    assert torch.equal(bgrad.cpu(), reduced.cpu())
    assert torch.equal(casted.cpu().t().to(dtype), casted_t.cpu().to(dtype))

    if stochastic:
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=rtol, atol=0.0)
    assert amax.cpu()[1][2] == torch.max(input.abs())


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_cast_transpose_bgrad_optional(shape, dtype, is_scale, is_amax):
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    input_hpu = input.to(hpu)

    scale_val = 0.5 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    amax_tensor = amax[1][2] if is_amax else None
    bgrad, casted, casted_t = fp8_cast_transpose_bgrad_fused(input_hpu, scale_hpu, amax_tensor, False)

    reduced = torch.sum(input_hpu, 0)
    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype)

    assert torch.equal(bgrad.cpu(), reduced.cpu())
    assert torch.equal(casted.cpu().t(), casted_t.cpu())

    rtol = 0.01 if dtype == torch.bfloat16 else 0.0
    assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=rtol, atol=0.0)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("scale", [1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("retain", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_fp8_cast_transpose_bgrad_dgelu(shape, scale, dtype, stochastic, retain, out_dtype):
    check_native_fp8(out_dtype)
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

    scale = torch.tensor(scale, dtype=torch.float)
    scale_inv = scale.reciprocal()

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    retain_tensor = None
    if retain:
        retain_tensor = (
            torch.tanh(torch.sqrt(torch.tensor(2 / np.pi, dtype=dtype)) * (input + 0.044715 * torch.pow(input, 3)))
            .to(dtype)
            .to(hpu)
        )
    bgrad, casted, casted_t = fp8_cast_transpose_bgrad_dgelu_fused(
        grad_hpu,
        input_hpu,
        scale.to(hpu),
        amax[1][2],
        stochastic,
        retain_tensor,
        out_dtype,
    )

    gelu = torch.nn.GELU(approximate="tanh")
    gelu_res = gelu(input)
    gelu_bwd = gelu_res.grad_fn(grad)
    reduced = torch.sum(gelu_bwd, 0)

    scaled_input_low_precision = simulateFp8Precision(gelu_bwd * scale, out_dtype)
    unscaled_input = scaled_input_low_precision * scale_inv

    uncasted = cast_from_fp8(casted, scale_inv.to(hpu), dtype).cpu()

    assert torch.allclose(bgrad.cpu(), reduced)
    assert torch.equal(casted.cpu().t().to(dtype), casted_t.cpu().to(dtype))

    if stochastic:
        assert torch.allclose(uncasted, unscaled_input, rtol=0.26, atol=0.01)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(uncasted, unscaled_input, rtol=rtol, atol=0.01)
    assert amax.cpu()[1][2] == torch.max(gelu_bwd.abs())


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("retain", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_cast_transpose_bgrad_dgelu_optional(shape, dtype, retain, is_scale, is_amax):
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

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    retain_tensor = None
    if retain:
        retain_tensor = (
            torch.tanh(torch.sqrt(torch.tensor(2 / np.pi, dtype=dtype)) * (input + 0.044715 * torch.pow(input, 3)))
            .to(dtype)
            .to(hpu)
        )
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    amax_tensor = amax[1][2] if is_amax else None
    bgrad, casted, casted_t = fp8_cast_transpose_bgrad_dgelu_fused(
        grad_hpu, input_hpu, scale_hpu, amax_tensor, False, retain_tensor
    )

    gelu = torch.nn.GELU(approximate="tanh")
    gelu_res = gelu(input)
    gelu_bwd = gelu_res.grad_fn(grad)
    reduced = torch.sum(gelu_bwd, 0)

    scaled_input_low_precision = simulateFp8Precision(gelu_bwd * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype).cpu()

    assert torch.allclose(bgrad.cpu(), reduced)
    assert torch.equal(casted.cpu().t(), casted_t.cpu())
    rtol = 0.01 if dtype == torch.bfloat16 else 0.0
    assert torch.allclose(uncasted, unscaled_input, rtol=rtol, atol=0.0)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(gelu_bwd.abs())


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("scale", [0.75])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_fp8_gelu(shape, scale, dtype, stochastic, is_scale, is_amax, out_dtype):
    check_native_fp8(out_dtype)
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
    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    amax_tensor = amax[1][2] if is_amax else None
    retain = torch.empty((shape[0] * 2, shape[1]), dtype=dtype).to(hpu)
    gelu_scaled = fp8_gelu(input.to(hpu), scale_hpu, amax_tensor, stochastic, retain, out_dtype)
    gelu_unscaled = cast_from_fp8(gelu_scaled, scale_inv_hpu, dtype).cpu()

    if stochastic:
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=0.26, atol=0.01)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=rtol, atol=0.0)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())
    assert torch.equal(retain.cpu(), retain_cpu)


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_fp8_gelu_v2(shape, scale, dtype, stochastic, is_scale, is_amax, out_dtype):
    check_native_fp8(out_dtype)
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
    gelu_unscaled = cast_from_fp8(gelu_scaled, scale_inv_hpu, dtype).cpu()

    if stochastic:
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=0.26, atol=0.01)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=rtol, atol=0.0)
    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
    assert torch.equal(retain.cpu(), retain_cpu)


@pytest.mark.parametrize("shape", [(32, 48)])
@pytest.mark.parametrize("scale", [0.75])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_fp8_fast_softmax(shape, scale, dtype, stochastic, is_scale, is_amax, out_dtype):
    check_native_fp8(out_dtype)
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input = ((torch.rand(shape, dtype=dtype) - 0.5) * 5).to("hpu")
    mask = torch.randint(0, 2, shape, dtype=torch.int).to(torch.bfloat16).to("hpu")
    scale_softmax = 0.17

    scale_val = scale if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()

    softmax_ref = torch.ops.hpu.scaled_masked_softmax(input, mask, scale_softmax).cpu()
    softmax_ref_low_precision = simulateFp8Precision(softmax_ref * scale, out_dtype)
    result_cpu = softmax_ref_low_precision * scale_inv

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None

    softmax, amax = torch.ops.hpu.fp8_fast_softmax(
        input, mask, scale_hpu, scale_softmax, stochastic, is_amax, out_dtype
    )
    softmax_unscaled = cast_from_fp8(softmax, scale_inv_hpu, dtype).cpu()

    if stochastic:
        assert torch.allclose(softmax_unscaled, result_cpu, rtol=0.26, atol=0.01)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(softmax_unscaled, result_cpu, rtol=rtol, atol=0.01)
    if is_amax:
        assert amax.cpu() == torch.max(softmax_ref.abs())


@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("retain", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_fp8_bgrad_dgelu_optional(shape, dtype, retain, is_scale, is_amax, out_dtype):
    check_native_fp8(out_dtype)
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

    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype).cpu()

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
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_fp8_dropout(shape, scale, dtype, is_scale, is_amax, out_dtype):
    check_native_fp8(out_dtype)
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
    dropout_unscaled = cast_from_fp8(dropout_scaled, scale_inv.to(hpu), dtype).cpu()

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


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("scale", [1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", FP8_NAMES_LEGACY)
def test_layernorm_fp8_fwd(shape, scale, dtype, stochastic, is_scale, is_amax, out_dtype):
    check_native_fp8(out_dtype)
    out_dtype = dtype_from_string(out_dtype)
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0] * 2, shape[1])

    weight = torch.rand((full_shape[1],), dtype=dtype)
    bias = torch.rand((full_shape[1],), dtype=dtype)
    eps = 0.07

    scale_val = scale if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scale_inv = scale.reciprocal()

    norm_cpu, mean_cpu, rstd_cpu = torch.native_layer_norm(input, (full_shape[1],), weight, bias, eps)
    mean_cpu = mean_cpu.reshape((full_shape[0],))
    rstd_cpu = rstd_cpu.reshape((full_shape[0],))

    scaled_norm_low_precision = simulateFp8Precision(norm_cpu * scale, out_dtype)
    result_norm_cpu = scaled_norm_low_precision * scale_inv

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    amax_tensor = amax[1][2] if is_amax else None
    norm_hpu, mean_hpu, rstd_hpu = layernorm_fwd_fp8(
        input.to(hpu),
        weight.to(hpu),
        bias.to(hpu),
        eps,
        scale_hpu,
        amax_tensor,
        stochastic,
        out_dtype,
    )
    norm_hpu_unscaled = cast_from_fp8(norm_hpu, scale_inv_hpu, dtype).cpu()

    rtol = 1e-3 if dtype == torch.float else 1e-1
    atol = 1e-3 if dtype == torch.float else 1e-1

    if is_amax:
        assert torch.allclose(
            amax.cpu()[1][2],
            torch.max(norm_cpu.to(torch.float).abs()),
            rtol=rtol,
            atol=atol,
        )
    assert torch.allclose(mean_hpu.cpu().to(dtype), mean_cpu, rtol=rtol, atol=atol)
    assert torch.allclose(rstd_hpu.cpu().to(dtype), rstd_cpu, rtol=rtol, atol=atol)

    result_atol = 0.01 if dtype == torch.float else 0.65
    if stochastic:
        assert torch.allclose(norm_hpu_unscaled, result_norm_cpu, rtol=0.26, atol=result_atol)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(norm_hpu_unscaled, result_norm_cpu, rtol=rtol, atol=result_atol)


@pytest.mark.parametrize(
    "shapeA, shapeB",
    [((2, 3, 4, 2), (2, 3, 4, 8)), ((64, 48), (64, 112))],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_tensor", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", FP8_NAMES_LEGACY)
def test_fp8_gemm(shapeA, shapeB, bias, out_tensor, accumulate, scaleA, scaleB, dtype, fp8_dtype):
    check_native_fp8(fp8_dtype)
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

    variant = variant_from_dtype(fp8_dtype)
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
    out_hpu = out.to(hpu) if out_tensor else None

    A8, _ = cast_to_fp8_v2(A_hpu, scaleA_hpu, False, False, fp8_dtype)
    B8, _ = cast_to_fp8_v2(B_hpu, scaleB_hpu, False, False, fp8_dtype)

    maybe_result = fp8_gemm(
        A8,
        scaleAInv,
        B8,
        scaleBInv,
        out_dtype=dtype,
        out=out_hpu,
        bias=bias_tensor_hpu,
        use_bias=bias,
        accumulate=accumulate,
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


@pytest.mark.parametrize("shape", [(2, 4)])
@pytest.mark.parametrize("is_out", [True, False])
def test_transpose(shape, is_out):
    torch.manual_seed(12345)
    hpu = torch.device("hpu")
    input = (torch.rand(shape) * 50).to(torch.int8)
    input_hpu = input.to(hpu)
    if is_out:
        out = torch.empty(shape[1], shape[0], dtype=torch.int8).to(hpu)
        fp8_transpose(input_hpu, out)
    else:
        out = fp8_transpose(input_hpu)
    out_ref = input.t()

    assert np.array_equal(out.cpu(), out_ref)


@pytest.mark.parametrize("shape", [(2, 2), (512,), (5, 4, 3, 8)])
def test_fp8_copy_(shape):
    torch.manual_seed(12345)
    hpu = torch.device("hpu")

    self = cast_to_fp8((torch.zeros(shape)).to(hpu))
    htcore.mark_step()
    src = cast_to_fp8((torch.randn(shape) * 50).to(hpu))

    torch.ops.hpu.fp8_copy_(self, src)
    assert np.array_equal(self.cpu(), src.cpu())


@pytest.mark.parametrize("shape", [(1, 4, 1, 32, 1), (4, 4, 8, 256, 32)])
def test_fp8_kv_reorder(shape):
    torch.manual_seed(12345)
    input_cpu = torch.rand(shape, dtype=torch.float32)
    start_cpu = torch.randint(0, 16, (shape[0],), dtype=torch.int32)
    end_cpu = torch.randint(0, 16, (shape[0],), dtype=torch.int32)
    beam_idx_cpu = torch.randint(0, 4, (shape[0], 4), dtype=torch.int32)

    input_hpu = cast_to_fp8(input_cpu.to(hpu))
    start_hpu = start_cpu.to(hpu)
    end_hpu = (start_cpu + end_cpu).to(hpu)
    beam_idx_hpu = torch.sum(beam_idx_cpu.to(hpu) * torch.tensor([[64, 16, 4, 1]]).to(hpu), axis=-1).to(torch.uint8)

    torch.ops.hpu.fp8_kv_reorder_(input_hpu, start_hpu, end_hpu, beam_idx_hpu)

    for i in range(shape[0]):
        subset = torch.narrow(input_cpu[i], -2, start_cpu[i], end_cpu[i])
        updated = subset.index_select(0, beam_idx_cpu[i])
        subset.copy_(updated)

    reference = cast_to_fp8(input_cpu.to(hpu)).cpu()
    np.testing.assert_equal(input_hpu.cpu().numpy(), reference.numpy())


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize("shape", [(5, 7), (6, 4, 8), (6, 4, 8, 12)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("is_full_shape", [True, False])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_index_copy(shape, dim, is_full_shape, dtype):
    torch.manual_seed(12345)
    self_tensor = torch.zeros(shape, dtype=dtype)
    self_tensor_h = cast_to_fp8(self_tensor.to("hpu"))
    dim_size = shape[dim]
    updates_shape = list(shape)

    if is_full_shape:
        idx = np.random.permutation(dim_size)
    else:
        updates_shape[dim] = dim_size - 2
        idx = np.random.choice(dim_size, size=[dim_size - 2], replace=False)

    updates_tensor = (
        torch.randint(low=-5, high=5, size=updates_shape, dtype=dtype)
        if dtype == torch.int
        else torch.randn(updates_shape, dtype=dtype)
    )
    updates_tensor_h = cast_to_fp8(updates_tensor.to("hpu"))
    index_tensor = torch.tensor(idx)
    index_tensor_h = index_tensor.to("hpu")

    self_tensor.index_copy_(dim, index_tensor, updates_tensor)
    htcore.mark_step()
    torch.ops.hpu.fp8_index_copy_(self_tensor_h, dim, index_tensor_h, updates_tensor_h)
    htcore.mark_step()
    self_tensor_h = cast_from_fp8(self_tensor_h, out_dtype=torch.float, scale=None)

    self_tensor = cast_from_fp8(cast_to_fp8(self_tensor.to("hpu")), out_dtype=torch.float, scale=None).to("cpu")
    compare_tensors(self_tensor_h, self_tensor, atol=0.0, rtol=0.0)


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize(
    "shape, repeats",
    [
        ([3], [2]),
        ([3], [2, 3]),
        ([3, 5], [2, 3]),
        ([3, 5], [2, 3, 4]),
        ([3, 5], [2, 3, 4, 5]),
        ([3, 5, 7], [2, 3, 4]),
        ([3, 5, 7], [2, 3, 4, 5]),
    ],
)
def test_hpu_repeat(shape, repeats):
    torch.manual_seed(12345)
    self = torch.rand(shape, dtype=torch.float)
    self_h = cast_to_fp8(self.to("hpu"))

    out = self.repeat(*repeats)
    out_h = torch.ops.hpu.fp8_repeat_v2(self_h, repeats)

    out_h = cast_from_fp8(out_h, out_dtype=torch.float, scale=None)
    out = cast_from_fp8(cast_to_fp8(out.to("hpu")), out_dtype=torch.float, scale=None).to("cpu")
    compare_tensors(out_h, out, atol=0.0, rtol=0.0)


@pytest.mark.skip(reason="Deprecated op")
@pytest.mark.parametrize(
    "shape, dim, index",
    [
        ([2, 3, 4], 0, [1]),
        ([2, 3, 4], 1, [1, 2]),
        ([2, 3, 4], 2, [0, 3]),
        ([2, 3, 4], -1, [0, 3]),
    ],
)
def test_hpu_index_select(shape, dim, index):
    self = torch.rand(shape, dtype=torch.float)
    self_h = cast_to_fp8(self.to("hpu"))
    index = torch.tensor(index, dtype=torch.int)
    index_h = index.to("hpu")

    out = torch.index_select(self, dim, index)
    out_h = torch.ops.hpu.fp8_index_select_v2(self_h, dim, index_h)

    out_h = cast_from_fp8(out_h, out_dtype=torch.float, scale=None)
    out = cast_from_fp8(cast_to_fp8(out.to("hpu")), out_dtype=torch.float, scale=None).to("cpu")
    compare_tensors(out_h, out, atol=0.0, rtol=0.0)
