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
from test_utils import cpu, hpu, is_gaudi1, compare_tensors
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels.Fp8Ops import cast_to_fp8, cast_to_fp8_v2, fp8_gemm, fp8_gemm_v2, fp8_transpose, cast_from_fp8, fp8_gelu, fp8_cast_transpose_fused, fp8_cast_transpose_bgrad_fused, layernorm_fwd_fp8, fp8_cast_transpose_bgrad_dgelu_fused

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht
import habana_frameworks.torch.core as htcore
ht.disable_dynamic_shape()

pytestmark = pytest.mark.skipif(is_gaudi1(), reason="Gaudi1 doesn't support fp8")

MASK_FLOAT32 = torch.tensor(2145386496, dtype=torch.int) # 0 11111111 11000000000000000000000b
MASK_ROUND_FLOAT32 = torch.tensor(1048575, dtype=torch.int) # 0 00000000 00011111111111111111111b
MASK_BFLOAT16 = torch.tensor(32736, dtype=torch.short) # 0 11111111 1100000b
MASK_ROUND_BFLOAT16 = torch.tensor(15, dtype=torch.short) # 0 00000000 0001111b
FP8_MAX = torch.tensor(57344*0.9, dtype=torch.float)

def simulateFp8Precision(input):
    dtype = input.dtype
    if dtype == torch.float:
        int_type = torch.int
        mask = MASK_FLOAT32
        mask_round = MASK_ROUND_FLOAT32
        excessive_bits = torch.tensor(21, dtype=int_type)
    else:
        int_type = torch.short
        mask = MASK_BFLOAT16
        mask_round = MASK_ROUND_BFLOAT16
        excessive_bits = torch.tensor(5, dtype=int_type)
    signs = torch.where(input < 0.0, -1.0, 1.0).to(dtype)
    asInt = input.view(int_type)
    mant_odd = torch.bitwise_and(torch.bitwise_right_shift(asInt, excessive_bits), torch.tensor(1, dtype=int_type))
    asInt_masked = asInt + mask_round
    asInt_odded = asInt_masked + mant_odd
    masked = torch.bitwise_and(asInt_odded, mask)
    return masked.view(dtype)*signs

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(72, 56, 16), (64, 48)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("transposed, allocate_out", [(True, True), (True, False), (False, False)])
def test_cast_to_fp8(shape, scale, dtype, stochastic, transposed, allocate_out):
    if transposed and len(shape) != 2:
        pytest.skip("Transpose cast supports only 2D tensors")
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0]*2, shape[1])

    scale = torch.tensor(scale, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    if transposed:
        if allocate_out:
            casted = torch.empty(
                full_shape,
                dtype=torch.int8,
                device="hpu",
            )
            casted_t = torch.empty(
                (full_shape[1], full_shape[0]),
                dtype=torch.int8,
                device="hpu",
            )
            fp8_cast_transpose_fused(input.to(hpu), scale.to(hpu), amax[1][2], stochastic, casted, casted_t)
        else:
            casted, casted_t = fp8_cast_transpose_fused(input.to(hpu), scale.to(hpu), amax[1][2], stochastic)
        assert torch.equal(casted.cpu().t(), casted_t.cpu())
    else:
        casted = cast_to_fp8(input.to(hpu), scale.to(hpu), amax[1][2], stochastic)
    uncasted = cast_from_fp8(casted, scale_inv.to(hpu), dtype)

    if stochastic:
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0)
    else:
        assert torch.equal(uncasted.cpu(), unscaled_input)
    assert amax.cpu()[1][2] == torch.max(input.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(72, 56, 16), (64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
def test_cast_to_fp8_v2(shape, dtype, stochastic, is_amax, is_scale):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0]*2, shape[1])

    scale_val = 1.3 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    casted, amax = cast_to_fp8_v2(input.to(hpu), scale_hpu, stochastic, is_amax)
    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype)

    if stochastic:
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0)
    else:
        assert torch.equal(uncasted.cpu(), unscaled_input)

    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
    else:
        assert amax.numel() == 0

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(72, 56)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_cast_to_fp8_transpose_optional(shape, dtype, is_scale, is_amax):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_val = 1.3 if is_scale else 1.0
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
    assert torch.equal(uncasted.cpu(), unscaled_input)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(72, 56)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_cast_to_fp8_optional(shape, dtype, is_scale, is_amax):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_val = 1.3 if is_scale else 1.0
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

    assert torch.equal(uncasted.cpu(), unscaled_input)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(64, 48), (6, 9)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
def test_fp8_cast_transpose_bgrad(shape, scale, dtype, stochastic):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    input_hpu = input.to(hpu)

    scale = torch.tensor(scale, dtype=torch.float)
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    bgrad, casted, casted_t = fp8_cast_transpose_bgrad_fused(input_hpu, scale.to(hpu), amax[1][2], stochastic)

    reduced = torch.sum(input_hpu, 0)
    uncasted = cast_from_fp8(casted, scale_inv.to(hpu), dtype)

    assert torch.equal(bgrad.cpu(), reduced.cpu())
    assert torch.equal(casted.cpu().t(), casted_t.cpu())

    if stochastic:
        assert torch.allclose(uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0)
    else:
        assert torch.equal(uncasted.cpu(), unscaled_input)
    assert amax.cpu()[1][2] == torch.max(input.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_cast_transpose_bgrad_optional(shape, dtype, is_scale, is_amax):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    input_hpu = input.to(hpu)

    scale_val = 1.3 if is_scale else 1.0
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

    assert torch.equal(uncasted.cpu(), unscaled_input)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(64, 48), (6, 9)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("retain", [True, False])
def test_fp8_cast_transpose_bgrad_dgelu(shape, scale, dtype, stochastic, retain):
    hpu = torch.device("hpu")
    full_shape = (shape[0]*2, shape[1])
    input_pos = torch.rand(shape, dtype=dtype, requires_grad=True)*30 + 10
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
        retain_tensor = torch.tanh(torch.sqrt(torch.tensor(2/np.pi, dtype=dtype))*(input +  0.044715*torch.pow(input, 3))).to(dtype).to(hpu)
    bgrad, casted, casted_t = fp8_cast_transpose_bgrad_dgelu_fused(grad_hpu, input_hpu, scale.to(hpu), amax[1][2], stochastic, retain_tensor)

    gelu = torch.nn.GELU(approximate='tanh')
    gelu_res = gelu(input)
    gelu_bwd = gelu_res.grad_fn(grad)
    reduced = torch.sum(gelu_bwd, 0)

    scaled_input_low_precision = simulateFp8Precision(gelu_bwd * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    uncasted = cast_from_fp8(casted, scale_inv.to(hpu), dtype).cpu()

    assert torch.allclose(bgrad.cpu(), reduced)
    assert torch.equal(casted.cpu().t(), casted_t.cpu())

    if stochastic:
        assert torch.allclose(uncasted, unscaled_input, rtol=0.26, atol=0.01)
    else:
        assert torch.allclose(uncasted, unscaled_input, rtol=0.0, atol=0.01)
    assert amax.cpu()[1][2] == torch.max(gelu_bwd.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("retain", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_cast_transpose_bgrad_dgelu_optional(shape, dtype, retain, is_scale, is_amax):
    hpu = torch.device("hpu")
    full_shape = (shape[0]*2, shape[1])
    input_pos = torch.rand(shape, dtype=dtype, requires_grad=True)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    input_hpu = input.to(hpu)
    grad = torch.rand(full_shape, dtype=dtype)
    grad_hpu = grad.to(hpu)

    scale_val = 1.3 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()

    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    retain_tensor = None
    if retain:
        retain_tensor = torch.tanh(torch.sqrt(torch.tensor(2/np.pi, dtype=dtype))*(input +  0.044715*torch.pow(input, 3))).to(dtype).to(hpu)
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    amax_tensor = amax[1][2] if is_amax else None
    bgrad, casted, casted_t = fp8_cast_transpose_bgrad_dgelu_fused(grad_hpu, input_hpu, scale_hpu, amax_tensor, False, retain_tensor)

    gelu = torch.nn.GELU(approximate='tanh')
    gelu_res = gelu(input)
    gelu_bwd = gelu_res.grad_fn(grad)
    reduced = torch.sum(gelu_bwd, 0)

    scaled_input_low_precision = simulateFp8Precision(gelu_bwd * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype).cpu()

    assert torch.allclose(bgrad.cpu(), reduced)
    assert torch.equal(casted.cpu().t(), casted_t.cpu())
    assert torch.allclose(uncasted, unscaled_input, rtol=0.0, atol=0.01)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(gelu_bwd.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(64, 48), (3, 4)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_gelu(shape, scale, dtype, stochastic, is_scale, is_amax):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_val = scale if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    gelu = torch.nn.GELU(approximate='tanh')
    gelu_res = gelu(input)
    scaled_gelu_low_precision = simulateFp8Precision(gelu_res * scale)
    result_cpu = scaled_gelu_low_precision * scale_inv
    retain_cpu = torch.tanh(torch.sqrt(torch.tensor(2/np.pi, dtype=dtype))*(input +  0.044715*torch.pow(input, 3))).to(dtype)

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    amax_tensor = amax[1][2] if is_amax else None
    retain = torch.empty((shape[0]*2, shape[1]), dtype=dtype).to(hpu)
    gelu_scaled = fp8_gelu(input.to(hpu), scale_hpu, amax_tensor, stochastic, retain)
    gelu_unscaled = cast_from_fp8(gelu_scaled, scale_inv_hpu, dtype).cpu()

    if stochastic:
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=0.26, atol=0.01)
    else:
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=0.0, atol=0.01)
    if is_amax:
        assert amax.cpu()[1][2] == torch.max(input.abs())
    assert torch.equal(retain.cpu(), retain_cpu)

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(64, 48), (3, 4)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_gelu_v2(shape, scale, dtype, stochastic, is_scale, is_amax):
    if is_amax is False and is_scale is True and stochastic is True and dtype == torch.float and scale == 0.75 and shape == (64, 48):
        pytest.xfail(reason="")

    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_val = scale if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()
    gelu = torch.nn.GELU(approximate='tanh')
    gelu_res = gelu(input)
    scaled_gelu_low_precision = simulateFp8Precision(gelu_res * scale)
    result_cpu = scaled_gelu_low_precision * scale_inv
    retain_cpu = torch.tanh(torch.sqrt(torch.tensor(2/np.pi, dtype=dtype))*(input +  0.044715*torch.pow(input, 3))).to(dtype)

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    gelu_scaled, retain, amax = torch.ops.hpu.fp8_gelu_v2(input.to(hpu), scale_hpu, stochastic, is_amax)
    gelu_unscaled = cast_from_fp8(gelu_scaled, scale_inv_hpu, dtype).cpu()

    print(gelu_unscaled.dtype)
    print(result_cpu.dtype)

    if stochastic:
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=0.26, atol=0.01)
    else:
        assert torch.allclose(gelu_unscaled.cpu(), result_cpu, rtol=0.0, atol=0.01)
    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
    assert torch.equal(retain.cpu(), retain_cpu)

@pytest.mark.xfail(reason="synNodeCreateWithId failed for node: fp8_fast_softmax_bf16 with synStatus 26 [Generice failure].")
@pytest.mark.parametrize("shape", [(96, 128), (3, 4)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_fast_softmax(shape, scale, dtype, stochastic, is_scale, is_amax):
    hpu = torch.device("hpu")
    input = ((torch.rand(shape, dtype=dtype) - 0.5) * 5).to("hpu")
    mask = torch.randint(0, 2, shape, dtype=torch.int).to(torch.bfloat16).to("hpu")
    scale_softmax = 0.17

    scale_val = scale if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()

    softmax_ref = torch.ops.hpu.scaled_masked_softmax(input, mask, scale_softmax).cpu()
    softmax_ref_low_precision = simulateFp8Precision(softmax_ref * scale)
    result_cpu = softmax_ref_low_precision * scale_inv

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None

    softmax, amax = torch.ops.hpu.fp8_fast_softmax(input, mask, scale_hpu, scale_softmax, stochastic, is_amax)
    softmax_unscaled = cast_from_fp8(softmax, scale_inv_hpu, dtype).cpu()

    if stochastic:
        assert torch.allclose(softmax_unscaled, result_cpu, rtol=0.26, atol=0.01)
    else:
        assert torch.allclose(softmax_unscaled, result_cpu, rtol=0.0, atol=0.01)
    if is_amax:
        assert amax.cpu() == torch.max(softmax_ref.abs())

@pytest.mark.xfail
@pytest.mark.parametrize("shape", [(64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("retain", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_bgrad_dgelu_optional(shape, dtype, retain, is_scale, is_amax):
    hpu = torch.device("hpu")
    full_shape = (shape[0]*2, shape[1])
    input_pos = torch.rand(shape, dtype=dtype, requires_grad=True)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    input_hpu = input.to(hpu)
    grad = torch.rand(full_shape, dtype=dtype)
    grad_hpu = grad.to(hpu)

    scale_val = 1.3 if is_scale else 1.0
    scale = torch.tensor(scale_val, dtype=torch.float)
    scale_inv = scale.reciprocal()

    retain_tensor = None
    if retain:
        retain_tensor = torch.tanh(torch.sqrt(torch.tensor(2/np.pi, dtype=dtype))*(input +  0.044715*torch.pow(input, 3))).to(dtype).to(hpu)
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    casted, bgrad, amax = torch.ops.hpu.fp8_bgrad_dgelu(grad_hpu, input_hpu, scale_hpu, retain_tensor, False, is_amax)

    gelu = torch.nn.GELU(approximate='tanh')
    gelu_res = gelu(input)
    gelu_bwd = gelu_res.grad_fn(grad)
    reduced = torch.sum(gelu_bwd, 0)

    scaled_input_low_precision = simulateFp8Precision(gelu_bwd * scale)
    unscaled_input = scaled_input_low_precision * scale_inv

    uncasted = cast_from_fp8(casted, scale_inv_hpu, dtype).cpu()

    assert torch.allclose(bgrad.cpu(), reduced)
    assert torch.allclose(uncasted, unscaled_input, rtol=0.0, atol=0.01)
    if is_amax:
        assert amax.cpu() == torch.max(gelu_bwd.abs())

# TODO analyze why single elements of outputs differ for torch.bfloat16
@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("shape", [(64, 96)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_fp8_dropout(shape, scale, dtype, is_scale, is_amax):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    ratio = 0.3
    dropout_scale = torch.tensor(1.0/(1.0 - ratio), dtype=dtype)

    scale = torch.tensor(scale, dtype=torch.float) if is_scale else torch.tensor(1.0)
    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv = scale.reciprocal()

    dropout_scaled, mask, amax = torch.ops.hpu.fp8_dropout(input.to(hpu), ratio, scale_hpu, False, is_amax)
    dropout_unscaled = cast_from_fp8(dropout_scaled, scale_inv.to(hpu), dtype).cpu()

    scaled_input_low_precision = simulateFp8Precision(input*scale.to(dtype)*dropout_scale)*scale_inv.to(dtype)
    result_ref = torch.where(mask.cpu().to(torch.bool), scaled_input_low_precision, 0.0)

    scaled_input_high_prec = input * dropout_scale
    dropout_high_prec = torch.where(mask.cpu().to(torch.bool), scaled_input_high_prec, 0.0)

    ones = torch.count_nonzero(mask.cpu())
    ratio_res = 1.0 - ones / mask.numel()

    assert torch.allclose(result_ref, dropout_unscaled)
    if is_amax:
        assert amax.cpu() == torch.max(dropout_high_prec.abs())
    assert torch.isclose(ratio_res, torch.tensor(ratio), rtol=0.1, atol=0.1)

@pytest.mark.xfail(reason="Results mismatch")
@pytest.mark.parametrize("shape", [(64, 48), (2, 7)])
@pytest.mark.parametrize("scale", [0.75, 1.6])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_layernorm_fp8_fwd(shape, scale, dtype, stochastic, is_scale, is_amax):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype)*30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))
    full_shape = (shape[0]*2, shape[1])

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

    scaled_norm_low_precision = simulateFp8Precision(norm_cpu * scale)
    result_norm_cpu = scaled_norm_low_precision * scale_inv

    scale_hpu = scale.to(hpu) if is_scale else None
    scale_inv_hpu = scale_inv.to(hpu) if is_scale else None
    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    amax_tensor = amax[1][2] if is_amax else None
    norm_hpu, mean_hpu, rstd_hpu = layernorm_fwd_fp8(input.to(hpu), weight.to(hpu), bias.to(hpu), eps, scale_hpu, amax_tensor, stochastic)
    norm_hpu_unscaled = cast_from_fp8(norm_hpu, scale_inv_hpu, dtype).cpu()

    rtol = 1e-3 if dtype == torch.float else 1e-1
    atol = 1e-3 if dtype == torch.float else 1e-1

    if is_amax:
        assert torch.allclose(amax.cpu()[1][2], torch.max(norm_cpu.to(torch.float).abs()), rtol=rtol, atol=atol)
    assert torch.allclose(mean_hpu.cpu().to(dtype), mean_cpu, rtol=rtol, atol=atol)
    assert torch.allclose(rstd_hpu.cpu().to(dtype), rstd_cpu, rtol=rtol, atol=atol)

    result_atol = 0.01 if dtype == torch.float else 0.65
    if stochastic:
        assert torch.allclose(norm_hpu_unscaled, result_norm_cpu, rtol=0.26, atol=result_atol)
    else:
        assert torch.allclose(norm_hpu_unscaled, result_norm_cpu, rtol=0.0, atol=result_atol)

@pytest.mark.parametrize("shapeA, shapeB", [((2, 3, 4, 2), (2, 3, 4, 8)),
                                            ((5, 10, 6), (5, 10, 18)),
                                            ((64, 48), (64, 112))])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_tensor", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_gemm(shapeA, shapeB, bias, out_tensor, accumulate, scaleA, scaleB, dtype):
    if accumulate and not out_tensor:
        pytest.skip("Accumulate not supported without out_tensor")

    hpu = torch.device("hpu")
    A = torch.rand(shapeA, dtype=dtype)*10 + 30.0
    A_hpu = A.to(hpu)
    max_A = torch.max(torch.abs(A)).to(torch.float)

    B = torch.rand(shapeB, dtype=dtype)*10 + 30.0
    B_hpu = B.to(hpu)
    max_B = torch.max(torch.abs(B)).to(torch.float)

    scaleA_hpu = None
    scaleB_hpu = None
    scaleAInv = None
    scaleBInv = None

    if scaleA:
        scaleA_hpu = (FP8_MAX / max_A).to(hpu)
        scaleAInv = torch.reciprocal(scaleA_hpu)

    if scaleB:
        scaleB_hpu = (FP8_MAX / max_B).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)

    rank = len(shapeA)
    out_shape = shapeA[0:(rank-2)] + (shapeA[-1],) + (shapeB[-1],)
    bias_tensor = torch.rand(out_shape, dtype=dtype)*10 + 30.0
    bias_tensor_hpu = bias_tensor.to(hpu) if bias else None

    out = torch.full(out_shape, 1000.0, dtype=dtype)
    out_hpu = out.to(hpu) if out_tensor else None

    A8 = cast_to_fp8(A_hpu, scaleA_hpu, None, False)
    B8 = cast_to_fp8(B_hpu, scaleB_hpu, None, False)

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

@pytest.mark.parametrize("shapeA, shapeB", [((2, 3, 4, 2), (2, 3, 4, 8)),
                                            ((5, 10, 6), (5, 10, 18)),
                                            ((64, 48), (64, 112))])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_gemm_v2(shapeA, shapeB, bias, accumulate, scaleA, scaleB, dtype):
    hpu = torch.device("hpu")
    A = torch.rand(shapeA, dtype=dtype)*10 + 30.0
    A_hpu = A.to(hpu)
    max_A = torch.max(torch.abs(A)).to(torch.float)

    B = torch.rand(shapeB, dtype=dtype)*10 + 30.0
    B_hpu = B.to(hpu)
    max_B = torch.max(torch.abs(B)).to(torch.float)

    scaleA_hpu = None
    scaleB_hpu = None
    scaleAInv = None
    scaleBInv = None

    if scaleA:
        scaleA_hpu = (FP8_MAX / max_A).to(hpu)
        scaleAInv = torch.reciprocal(scaleA_hpu)

    if scaleB:
        scaleB_hpu = (FP8_MAX / max_B).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)

    rank = len(shapeA)
    out_shape = shapeA[0:(rank-2)] + (shapeA[-1],) + (shapeB[-1],)
    bias_tensor = torch.rand(out_shape, dtype=dtype)*10 + 30.0
    bias_tensor_hpu = bias_tensor.to(hpu) if bias else None

    out = torch.full(out_shape, 1000.0, dtype=dtype)
    out_hpu = out.to(hpu)

    A8 = cast_to_fp8(A_hpu, scaleA_hpu, None, False)
    B8 = cast_to_fp8(B_hpu, scaleB_hpu, None, False)

    result = fp8_gemm_v2(A8, scaleAInv, B8, scaleBInv, out_dtype=dtype, bias=bias_tensor_hpu, use_bias=bias, accumulate=accumulate, accumulate_to=out_hpu)
    result_ref = torch.matmul(A.transpose(-2, -1), B)

    if bias:
        result_ref = result_ref + bias_tensor
    if accumulate:
        result_ref = result_ref + out
    result = result.cpu()

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


@pytest.mark.parametrize("shape", [(2, 2), (512,), (5, 4, 3, 8)])
def test_fp8_copy_(shape):
    torch.manual_seed(0)
    hpu = torch.device("hpu")

    self = cast_to_fp8((torch.zeros(shape)).to(hpu))
    htcore.mark_step()
    src = cast_to_fp8((torch.randn(shape)*50).to(hpu))

    torch.ops.hpu.fp8_copy_(self, src)
    assert np.array_equal(self.cpu(), src.cpu())


@pytest.mark.parametrize("shape", [(1, 4, 1, 32, 1), (4, 4, 8, 256, 32)])
def test_fp8_kv_reorder(shape):
    torch.manual_seed(0)
    input_cpu = torch.rand(shape, dtype=torch.float32)
    start_cpu = torch.randint(0, 16, (shape[0],), dtype=torch.int32)
    end_cpu = torch.randint(0, 16, (shape[0],), dtype=torch.int32)
    beam_idx_cpu = torch.randint(0, 4, (shape[0], 4), dtype=torch.int32)

    input_hpu = cast_to_fp8(input_cpu.to(hpu))
    start_hpu = start_cpu.to(hpu)
    end_hpu = (start_cpu + end_cpu).to(hpu)
    beam_idx_hpu = torch.sum(
        beam_idx_cpu.to(hpu) * torch.tensor([[64, 16, 4, 1]]).to(hpu), axis=-1
    ).to(torch.uint8)

    torch.ops.hpu.fp8_kv_reorder_(input_hpu, start_hpu, end_hpu, beam_idx_hpu)

    for i in range(shape[0]):
        subset = torch.narrow(input_cpu[i], -2, start_cpu[i], end_cpu[i])
        updated = subset.index_select(0, beam_idx_cpu[i])
        subset.copy_(updated)

    reference = cast_to_fp8(input_cpu.to(hpu)).cpu()
    np.testing.assert_equal(input_hpu.cpu().numpy(), reference.numpy())
    assert np.array_equal(self.cpu(), src.cpu())

@pytest.mark.parametrize("shape", [(5, 7), (6, 4, 8), (6, 4, 8, 12)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("is_full_shape", [True, False])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_index_copy(shape, dim, is_full_shape, dtype):
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
