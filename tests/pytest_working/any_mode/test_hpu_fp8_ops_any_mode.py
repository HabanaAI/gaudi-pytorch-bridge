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
from enum import Enum

import numpy as np
import pytest
import torch
from fp8_utils import FP8_MAX, simulateFp8Precision
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
    is_pytest_mode_lazy,
)

Verbose = False

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht

ht.disable_dynamic_shape()

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support fp8")]

fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]


class ScaleMode(Enum):
    TENSOR = 1
    SCALAR = 2
    TENSOR_CHANNEL = 3
    SCALAR_CHANNEL = 4


@pytest.mark.parametrize("shape", [(64, 48)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize(
    "scale_mode, axis",
    [
        [ScaleMode.TENSOR, None],
        [ScaleMode.SCALAR, None],
        [ScaleMode.TENSOR_CHANNEL, 0],
        [ScaleMode.TENSOR_CHANNEL, 1],
        [ScaleMode.SCALAR_CHANNEL, 0],
        [ScaleMode.SCALAR_CHANNEL, 1],
        [None, None],
    ],
)
@pytest.mark.parametrize("out_dtype", fp8_dtypes, ids=format_tc)
def test_cast_to_fp8_v2(shape, dtype, stochastic, is_amax, scale_mode, axis, out_dtype):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_shape = None
    scale = torch.tensor(1.0)
    scale_hpu = None
    scale_inv_hpu = None

    if scale_mode:
        if scale_mode in [ScaleMode.TENSOR, ScaleMode.SCALAR]:
            scale_val = 1.3
        else:
            scale_val = (np.random.rand(input.shape[-1 - axis]) * 2.0 + 0.5).astype(np.float32)
        scale = torch.tensor(scale_val)

        if scale_mode in [ScaleMode.TENSOR, ScaleMode.TENSOR_CHANNEL]:
            scale_hpu = scale.to("hpu")
            scale_inv_hpu = scale.reciprocal().to("hpu")
        else:
            scale_hpu = scale_val
            scale_inv_hpu = 1 / scale_val
            if scale_mode == ScaleMode.SCALAR_CHANNEL:
                scale_hpu = scale_hpu.tolist()
                scale_inv_hpu = scale_inv_hpu.tolist()

        if scale_mode in [ScaleMode.TENSOR_CHANNEL, ScaleMode.SCALAR_CHANNEL]:
            scale = torch.unsqueeze(scale, axis)
            scale_shape = scale.shape

    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(input * scale.to(dtype), out_dtype)
    unscaled_input = scaled_input_low_precision * scale_inv.to(dtype)

    def fn(
        input,
        scale,
        scale_inv,
        stochastic,
        is_amax,
        out_dtype,
        dtype,
        scale_shape,
    ):
        args = [input, scale, stochastic, is_amax, out_dtype]
        if scale_shape is not None:
            args.append(scale_shape)
        casted, amax = torch.ops.hpu.cast_to_fp8_v2(*args)
        uncasted = torch.ops.hpu.cast_from_fp8(casted, scale_inv, dtype, scale_shape)
        return casted, amax, uncasted

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    casted, amax, uncasted = fn(
        input.to(hpu),
        scale_hpu,
        scale_inv_hpu,
        stochastic,
        is_amax,
        out_dtype,
        dtype,
        scale_shape,
    )

    uncasted_cpu = uncasted.cpu()

    if stochastic:
        assert torch.allclose(uncasted_cpu, unscaled_input, rtol=0.26, atol=0.0)
        assert not torch.equal(uncasted_cpu, unscaled_input)
    else:
        assert torch.equal(uncasted_cpu, unscaled_input)

    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
    else:
        assert amax.numel() == 0

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2", "cast_from_fp8"})


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("out_dtype", fp8_dtypes, ids=format_tc)
def test_cast_to_fp8_v2_out_of_range(dtype, stochastic, out_dtype):
    if out_dtype == torch.float8_e5m2:
        input = torch.tensor([100000, 60000, -60000, -100000], dtype=dtype).to("hpu")
        min = torch.finfo(out_dtype).min
        max = torch.finfo(out_dtype).max
    else:
        input = torch.tensor([1000, 300, -300, -1000], dtype=dtype).to("hpu")
        min = -240.0
        max = 240.0
    expected = torch.tensor([max, max, min, min], dtype=out_dtype).to("hpu")

    result, _ = torch.ops.hpu.cast_to_fp8_v2(input, None, stochastic, False, out_dtype)

    assert torch.equal(result, expected)


# casting bf16 to f8 uses SFTZ rounding mode, which applies
# stochastic rounding also when rounding number between
# 0.0 and f8 min denormal value.
@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-175380")
def test_sftz_rounding_mode():
    input_dtype = torch.bfloat16
    target_dtype = torch.float8_e5m2
    shape = (100, 100)
    min_subnormal = pow(2, -16)
    value = min_subnormal / 3.0

    input = torch.full(shape, value, dtype=input_dtype).to("hpu")
    result, _ = torch.ops.hpu.cast_to_fp8_v2(input, None, True, False, target_dtype)
    result_cpu = result.cpu().float()

    expected_results = torch.tensor((0.0, min_subnormal))
    assert torch.equal(result_cpu.unique(), expected_results)
    assert torch.allclose(torch.mean(result_cpu), torch.tensor(value), atol=0.0, rtol=0.1)


@pytest.mark.parametrize("shape", [(16, 24, 8), (64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("is_scale_152", [True, False])
@pytest.mark.parametrize("is_scale_143", [True, False])
def test_cast_to_fp8_hybrid(shape, dtype, stochastic, is_amax, is_scale_152, is_scale_143):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_152_val = 1.3 if is_scale_152 else 1.0
    scale_152 = torch.tensor(scale_152_val, dtype=torch.float)
    scale_152_inv = scale_152.reciprocal()
    scale_143_val = 0.7 if is_scale_143 else 1.0
    scale_143 = torch.tensor(scale_143_val, dtype=torch.float)
    scale_143_inv = scale_143.reciprocal()

    scaled_input_low_precision_152 = simulateFp8Precision(input * scale_152.to(dtype), torch.float8_e5m2)
    unscaled_input_152 = scaled_input_low_precision_152 * scale_152_inv.to(dtype)

    scaled_input_low_precision_143 = simulateFp8Precision(input * scale_143.to(dtype), torch.float8_e4m3fn)
    unscaled_input_143 = scaled_input_low_precision_143 * scale_143_inv.to(dtype)

    scale_152_hpu = scale_152.to(hpu) if is_scale_152 else None
    scale_152_inv_hpu = scale_152_inv.to(hpu) if is_scale_152 else None
    scale_143_hpu = scale_143.to(hpu) if is_scale_143 else None
    scale_143_inv_hpu = scale_143_inv.to(hpu) if is_scale_143 else None

    def fn(
        input,
        scale_152,
        scale_143,
        scale_152_inv,
        scale_143_inv,
        stochastic,
        is_amax,
        dtype,
    ):
        casted_152, casted_143, amax = torch.ops.hpu.cast_to_fp8_hybrid(
            input, scale_152, scale_143, stochastic, is_amax
        )
        # to prevent casts optimization
        casted_152 = casted_152 * 1.0
        casted_143 = casted_143 * 1.0
        uncasted_152 = torch.ops.hpu.cast_from_fp8(casted_152, scale_152_inv, dtype)
        uncasted_143 = torch.ops.hpu.cast_from_fp8(casted_143, scale_143_inv, dtype)

        return casted_152, casted_143, amax, uncasted_152, uncasted_143

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    casted_152, casted_143, amax, uncasted_152, uncasted_143 = fn(
        input.to(hpu),
        scale_152_hpu,
        scale_143_hpu,
        scale_152_inv_hpu,
        scale_143_inv_hpu,
        stochastic,
        is_amax,
        dtype,
    )

    if stochastic:
        assert torch.allclose(uncasted_152.cpu(), unscaled_input_152, rtol=0.26, atol=0.0)
        assert torch.allclose(uncasted_143.cpu(), unscaled_input_143, rtol=0.26, atol=0.0)
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(uncasted_152.cpu(), unscaled_input_152, rtol=rtol, atol=0.0)
        assert torch.allclose(uncasted_143.cpu(), unscaled_input_143, rtol=rtol, atol=0.0)

    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
    else:
        assert amax.numel() == 0

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_hybrid", "cast_from_fp8"})


scale_modes = [
    (ScaleMode.TENSOR, ScaleMode.TENSOR),
    (ScaleMode.TENSOR, None),
    (ScaleMode.SCALAR, ScaleMode.SCALAR),
    (ScaleMode.SCALAR, None),
    (None, ScaleMode.TENSOR),
    (None, ScaleMode.SCALAR),
    (None, ScaleMode.TENSOR_CHANNEL),
    (None, ScaleMode.SCALAR_CHANNEL),
    (None, None),
]


@pytest.mark.parametrize(
    "shapeA, shapeB",
    [
        ((2, 1, 4, 2), (1, 3, 4, 8)),
        ((24, 12), (24, 36)),
        ((2, 1, 4, 2), (4, 8)),
        ((4, 2), (3, 4, 8)),
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("scaleA, scaleB", scale_modes)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes, ids=format_tc)
def test_fp8_gemm_v2(shapeA, shapeB, bias, accumulate, scaleA, scaleB, dtype, fp8_dtype):
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
    if scaleA == ScaleMode.TENSOR:
        scaleA_hpu = (FP8_MAX[variant] / max_A).to(hpu)
        scaleAInv = torch.reciprocal(scaleA_hpu)
    elif scaleA == ScaleMode.SCALAR:
        scaleA_hpu = (FP8_MAX[variant] / max_A).item()
        scaleAInv = 1 / scaleA_hpu
        if not scaleB:
            scaleBInv = 1.0

    if scaleB == ScaleMode.TENSOR:
        scaleB_hpu = (FP8_MAX[variant] / max_B).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)
    elif scaleB == ScaleMode.SCALAR:
        scaleB_hpu = (FP8_MAX[variant] / max_B).item()
        scaleBInv = 1 / scaleB_hpu
        if not scaleA:
            scaleAInv = 1.0
    elif scaleB == ScaleMode.TENSOR_CHANNEL:
        scaleB_hpu = (FP8_MAX[variant] / max_B).expand(shapeB[-1]).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)
    elif scaleB == ScaleMode.SCALAR_CHANNEL:
        scaleB_h = (FP8_MAX[variant] / max_B).expand(shapeB[-1])
        scaleBInv = (1 / scaleB_h).numpy().tolist()
        scaleB_hpu = scaleB_h.numpy().tolist()
        if not scaleA:
            scaleAInv = [1.0]

    result_ref = torch.matmul(A.transpose(-2, -1), B)

    out_shape = result_ref.shape
    bias_tensor = torch.rand(out_shape, dtype=dtype) * 10 + 30.0
    bias_tensor_hpu = bias_tensor.to(hpu) if bias else None

    out = torch.full(out_shape, 1000.0, dtype=dtype)
    out_hpu = out.to(hpu)

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

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

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

    if bias:
        result_ref = result_ref + bias_tensor
    if accumulate:
        result_ref = result_ref + out
    result = result.cpu()

    percentage_diff = torch.abs((((result - result_ref) / result_ref) * 100).to(torch.int))
    assert np.amax(percentage_diff.numpy()) <= 15

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2", "fp8_gemm_v2"})


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-171898")
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((10,), (10,)),
        ((2, 10), (10,)),
        ((10,), (10, 2)),
        ((4, 8, 16), (16,)),
        ((8,), (2, 4, 8, 16)),
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes, ids=format_tc)
def test_fp8_gemm_v2_1d(shape_a, shape_b, bias, scale, transpose_a, transpose_b, fp8_dtype):
    hpu = torch.device("hpu")
    dtype = torch.bfloat16

    def generate_input(shape, transpose):
        input = (torch.rand(shape, dtype=dtype) * 10 + 30.0).to(fp8_dtype)
        if transpose:
            if len(shape) == 1:
                pytest.skip("Configuration not supported")
            input_hpu = input.transpose(-2, -1).to(hpu)
        else:
            input_hpu = input.to(hpu)

        return input.to(dtype), input_hpu

    A, A_hpu = generate_input(shape_a, transpose_a)
    B, B_hpu = generate_input(shape_b, transpose_b)

    scaleA = 1.0
    scaleB = 1.0
    scaleA_hpu = None
    scaleB_hpu = None

    if scale:
        scaleA = torch.tensor(3.14, dtype=dtype)
        scaleB = torch.tensor(0.75, dtype=dtype)
        scaleA_hpu = scaleA.to("hpu")
        scaleB_hpu = scaleB.to("hpu")

    result_ref = torch.matmul(A, B) * torch.mul(scaleA, scaleB)

    out_shape = result_ref.shape
    bias_tensor = torch.rand(out_shape, dtype=dtype) * 10 + 30.0
    bias_tensor_hpu = bias_tensor.to(hpu) if bias else None

    fn = torch.ops.hpu.fp8_gemm_v2

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(
        A_hpu,
        transpose_a,
        B_hpu,
        transpose_b,
        None,
        dtype,
        scaleA_hpu,
        scaleB_hpu,
        bias_tensor_hpu,
        False,
    ).cpu()

    if bias:
        result_ref = result_ref + bias_tensor

    percentage_diff = torch.abs((((result - result_ref) / result_ref) * 100).to(torch.int))
    assert np.amax(percentage_diff.numpy()) <= 15

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("fp8_gemm_v2")


@pytest.mark.parametrize("scale_mode", [ScaleMode.TENSOR_CHANNEL, ScaleMode.SCALAR_CHANNEL])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("in_dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16])
def test_fp8_gemm_v2_scale_shape(scale_mode, axis, in_dtype, out_dtype):
    shapeA = (12, 24)
    shapeB = (24, 36)

    def getInputAndScale(is_vector):
        shape = shapeB if is_vector else shapeA
        input_cpu = (torch.rand(shape, dtype=out_dtype) * 10 + 30.0).to(in_dtype).to(out_dtype)
        input_hpu = input_cpu.to(in_dtype).to("hpu")

        if not is_vector:
            scale_length = 1
        elif axis == 1:
            scale_length = shapeA[0]
        else:
            scale_length = shapeB[1]
        scale_array = ((np.random.rand(scale_length) * 100.0).astype(np.float32)).tolist()
        scale_tensor = torch.tensor(scale_array)
        scale_hpu = scale_tensor.to("hpu") if scale_mode == ScaleMode.TENSOR_CHANNEL else scale_array

        if is_vector:
            scale_tensor = torch.unsqueeze(scale_tensor, axis)

        return input_cpu, input_hpu, scale_tensor, scale_hpu

    A, A_hpu, scale_a, scale_a_hpu = getInputAndScale(False)
    B, B_hpu, scale_b, scale_b_hpu = getInputAndScale(True)

    scale_shape = scale_b.shape

    fn = torch.ops.hpu.fp8_gemm_v2

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(
        A_hpu,
        False,
        B_hpu,
        False,
        None,
        out_dtype,
        scale_a_hpu,
        scale_b_hpu,
        None,
        False,
        scale_shape,
    )

    result_ref = torch.matmul(A, B) * (scale_a * scale_b)

    compare_tensors(result, result_ref, atol=1e-3, rtol=1e-2)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("fp8_gemm_v2")


@pytest.mark.parametrize("scaleA", [16, 14])
@pytest.mark.parametrize("scaleB", [0.00390625, 0.23])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_fp8_gemm_v2_scalar_optimization(scaleA, scaleB, dtype):
    if scaleA == 16 and scaleB == 0.00390625 and is_pytest_mode_eager():
        pytest.skip("Configuration not supported in eager mode yet.")

    ht.enable_inference_mode()
    shapeA = (12, 24)
    shapeB = (24, 36)
    fp8_dtype = torch.float8_e4m3fn
    A = (torch.rand(shapeA, dtype=dtype) * 10 + 30.0).to(fp8_dtype).to(dtype)
    A_hpu = A.to(fp8_dtype).to("hpu")

    B = (torch.rand(shapeB, dtype=dtype) * 10 + 30.0).to(fp8_dtype).to(dtype)
    B_hpu = B.to(fp8_dtype).to("hpu")

    fn = torch.ops.hpu.fp8_gemm_v2

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(
        A_hpu,
        False,
        B_hpu,
        False,
        None,
        dtype,
        scaleA,
        scaleB,
        None,
        False,
    )

    result_ref = torch.matmul(A, B) * (scaleA * scaleB)

    compare_tensors(result, result_ref, atol=1e-3, rtol=1e-2)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("fp8_gemm_v2")
    ht.disable_inference_mode()


@pytest.mark.skipif(is_pytest_mode_eager(), reason="Not supported in eager mode yet.")
@pytest.mark.parametrize("scale_a", [16.0, 1.0, 0.0625, 0.00390625])
@pytest.mark.parametrize("scale_b", [16.0, 1.0, 0.0625, 0.00390625])
@pytest.mark.parametrize("scale_out", [16.0, 1.0, 0.0625, 256.0])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_fp8_gemm_v2_bias_optimization(scale_a, scale_b, scale_out, dtype):
    ht.enable_inference_mode()

    a = (torch.rand(4, 8) * 5).to(torch.float8_e4m3fn).to("hpu")
    b = (torch.rand(8, 12) * 5).to(torch.float8_e4m3fn).to("hpu")

    scale_a_t = torch.tensor(scale_a).to("hpu")
    scale_b_t = torch.tensor(scale_b).to("hpu")
    scale_out_t = torch.tensor(scale_out).to("hpu")

    def fn(a, b, scale_a, scale_b, scale_out):
        return torch.ops.hpu.cast_to_fp8_v2(
            torch.ops.hpu.fp8_gemm_v2(a, False, b, False, None, dtype, scale_a, scale_b, None, False),
            scale_out,
            False,
            False,
            torch.float8_e4m3fn,
        )

    res_fp8_tensor, _ = fn(a, b, scale_a_t, scale_b_t, scale_out_t)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res_fp8_scalar, _ = fn(a, b, scale_a, scale_b, scale_out)
    res_scalar_cpu = res_fp8_scalar.cpu().float()

    rtol = 1e-3 if dtype == torch.float else 1e-2
    compare_tensors(res_fp8_tensor, res_scalar_cpu, atol=1e-2, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"fp8_gemm_v2", "cast_to_fp8_v2"})
    ht.disable_inference_mode()


@pytest.mark.skipif(not is_pytest_mode_lazy(), reason="Currently supported only in lazy mode.")
@pytest.mark.parametrize("scale_a", [16.0, 1.0, 7.5])
@pytest.mark.parametrize("scale_b", [16.0, 0.00390625, 7.5])
@pytest.mark.parametrize("scale_out", [0.0625, 256.0, 7.5])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_fp8_gemm_v2_mark_scales_const(scale_a, scale_b, scale_out, dtype):
    ht.enable_inference_mode()
    from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const

    def fn(a, b, scale_a, scale_b, scale_out):
        return torch.ops.hpu.cast_to_fp8_v2(
            torch.ops.hpu.fp8_gemm_v2(a, False, b, False, None, dtype, scale_a, scale_b, None, False),
            scale_out,
            False,
            False,
            torch.float8_e4m3fn,
        )

    class TestModel(torch.nn.Module):
        def __init__(self, input_scale, other_scale, out_scale):
            super(TestModel, self).__init__()
            self.input_scale = torch.nn.Parameter(input_scale)
            self.other_scale = torch.nn.Parameter(other_scale)
            self.out_scale = torch.nn.Parameter(out_scale)

        def forward(self, input, other):
            return fn(input, other, self.input_scale, self.other_scale, self.out_scale)

    a = (torch.rand(4, 8) * 5).to(torch.float8_e4m3fn).to("hpu")
    b = (torch.rand(8, 12) * 5).to(torch.float8_e4m3fn).to("hpu")

    scale_a_t = torch.tensor(scale_a, dtype=dtype).to("hpu")
    scale_b_t = torch.tensor(scale_b, dtype=dtype).to("hpu")
    scale_out_t = torch.tensor(scale_out).to("hpu")

    model = TestModel(scale_a_t, scale_b_t, scale_out_t)

    _mark_params_as_const(model)
    _check_params_as_const(model)

    res_fp8_scalar, _ = model(a, b)
    res_scalar_cpu = res_fp8_scalar.cpu().float()

    res_fp8_tensor, _ = fn(a, b, scale_a_t, scale_b_t, scale_out_t)

    rtol = 1e-3 if dtype == torch.float else 1e-2
    compare_tensors(res_fp8_tensor, res_scalar_cpu, atol=1e-2, rtol=rtol)
    ht.disable_inference_mode()


@pytest.mark.parametrize("shape", [(8, 2, 2, 5)])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + fp8_dtypes)
def test_in_place_interleave(shape, dtype):
    input = torch.randn(shape, dtype=torch.bfloat16) * 10.0
    input_hpu = input.to("hpu")
    if dtype != torch.bfloat16:
        input_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(input_hpu, None, False, False, dtype)
        input = simulateFp8Precision(input, dtype)

    indices = []
    for i in range(int(shape[0] / 4)):
        indices += [i] * 4
    index = torch.tensor(indices)

    def fn(input):
        torch.ops.hpu.in_place_interleave_(input)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    fn(input_hpu)

    if dtype != torch.bfloat16:
        input_hpu = torch.ops.hpu.cast_from_fp8(input_hpu, None, torch.bfloat16)

    output_ref = torch.index_select(input, 0, index)

    assert torch.equal(input_hpu.cpu(), output_ref)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("in_place_interleave")


@pytest.mark.parametrize("N, C, H, W", [(8, 3, 28, 28), (4, 6, 16, 16)])
@pytest.mark.parametrize("out_channels", [16])
@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("kernel", [(2, 2), (4, 6)], ids=format_tc)
@pytest.mark.parametrize("stride", [(1, 1), (2, 2)], ids=format_tc)
@pytest.mark.parametrize("padding", [(0, 0), (1, 1)], ids=format_tc)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes, ids=format_tc)
def test_conv2d_fp8(N, C, H, W, out_channels, scaleA, scaleB, kernel, stride, padding, bias, out_dtype, fp8_dtype):
    if pytest.mode == "eager" and kernel == (4, 6):
        pytest.skip("Configuration not supported")

    input_cpu = torch.rand((N, C, H, W), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    input_hpu = input_cpu.to("hpu").to(fp8_dtype)

    weight_cpu = torch.rand((out_channels, C, kernel[0], kernel[1]), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    weight_hpu = weight_cpu.to("hpu").to(fp8_dtype)

    bias_cpu = torch.rand(out_channels, dtype=out_dtype).to(fp8_dtype).to(out_dtype) if bias else None
    bias_hpu = bias_cpu.to("hpu") if bias else None

    conv_ref_unscaled = torch.nn.functional.conv2d(input_cpu, weight_cpu, None, stride, padding, 1, 1)

    def process_scale(scale, value):
        scale_cpu = 1
        scale_hpu = None
        if scale:
            scale_cpu = torch.tensor(value).to(out_dtype)
            scale_hpu = scale_cpu.to("hpu")
        return scale_cpu, scale_hpu

    scaleA_cpu, scaleA_hpu = process_scale(scaleA, 1.4)
    scaleB_cpu, scaleB_hpu = process_scale(scaleB, 2.3)

    if Verbose:
        print(f"{scaleA_cpu = }")
        print(f"{scaleA_hpu = }")
        print(f"{scaleB_cpu = }")
        print(f"{scaleB_hpu = }")

    fn = torch.ops.hpu.conv2d_fp8

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    conv_args = [input_hpu, weight_hpu, bias_hpu, stride, padding, 1, 1, out_dtype]
    if scaleA_hpu is not None or scaleB_hpu is not None:
        conv_args.extend([scaleA_hpu, scaleB_hpu])

    conv = fn(*conv_args)
    conv_ref = conv_ref_unscaled * (scaleA_cpu * scaleB_cpu)
    if bias_cpu is not None:
        bias_cpu = bias_cpu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        conv_ref = conv_ref + bias_cpu

    if out_dtype == torch.bfloat16 and (scaleA or scaleB):
        rtol = 0.02
    else:
        rtol = 1e-2

    compare_tensors(conv, conv_ref, atol=1e-2, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("conv2d_fp8")


@pytest.mark.parametrize("scaleA", [16, 14])
@pytest.mark.parametrize("scaleB", [0.00390625, 0.23])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes, ids=format_tc)
def test_conv2d_fp8_scalar_optimization(scaleA, scaleB, out_dtype, fp8_dtype):
    if scaleA == 16 and scaleB == 0.00390625:
        pytest.skip("Configuration supported only with PT_HPU_INFERENCE_MODE=1 flag.")
    N, C, H, W = (4, 3, 12, 12)
    out_channels = 16
    kernel = (2, 2)
    stride = (1, 1)
    padding = (0, 0)

    input_cpu = torch.rand((N, C, H, W), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    input_hpu = input_cpu.to("hpu").to(fp8_dtype)

    weight_cpu = torch.rand((out_channels, C, kernel[0], kernel[1]), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    weight_hpu = weight_cpu.to("hpu").to(fp8_dtype)

    fn = torch.ops.hpu.conv2d_fp8

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    conv = fn(input_hpu, weight_hpu, None, stride, padding, 1, 1, out_dtype, scaleA, scaleB)
    conv_ref = torch.nn.functional.conv2d(input_cpu, weight_cpu, None, stride, padding, 1, 1) * (scaleA * scaleB)

    compare_tensors(conv, conv_ref, atol=1e-2, rtol=0.02)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("conv2d_fp8")


# For manual testing with flag PT_HPU_INFERENCE_MODE=1
# Post synapse graphs should contain raw Conv node with exp_bias set on inputs and output.
@pytest.mark.parametrize("scale_a", [16.0, 1.0, 0.0625, 0.00390625])
@pytest.mark.parametrize("scale_b", [16.0, 1.0, 0.0625, 0.00390625])
@pytest.mark.parametrize("scale_out", [16.0, 1.0, 0.0625, 256.0])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16], ids=format_tc)
def DISABLED_test_conv2d_fp8_bias_optimization(scale_a, scale_b, scale_out, out_dtype):
    N, C, H, W = (4, 3, 12, 12)
    fp8_dtype = torch.float8_e4m3fn
    out_channels = 16
    kernel = (2, 2)
    stride = (1, 1)
    padding = (0, 0)

    input_hpu = (torch.rand(N, C, H, W) * 5).to(fp8_dtype).to("hpu")
    weight_hpu = (torch.rand(out_channels, C, kernel[0], kernel[1]) * 5).to(fp8_dtype).to("hpu")

    scale_a_t = torch.tensor(scale_a).to("hpu")
    scale_b_t = torch.tensor(scale_b).to("hpu")
    scale_out_t = torch.tensor(scale_out).to("hpu")

    def fn(input, weight, scale_a, scale_b, scale_out):
        return (
            torch.ops.hpu.cast_to_fp8_v2(
                torch.ops.hpu.conv2d_fp8(input, weight, None, stride, padding, 1, 1, out_dtype, scale_a, scale_b),
                scale_out,
                False,
                False,
                fp8_dtype,
            )[0]
            + 1.0
        )

    res_fp8_scalar = fn(input_hpu, weight_hpu, scale_a, scale_b, scale_out).cpu().float()
    res_fp8_tensor = fn(input_hpu, weight_hpu, scale_a_t, scale_b_t, scale_out_t).float()

    rtol = 1e-3 if out_dtype == torch.float else 0.125
    compare_tensors(res_fp8_tensor, res_fp8_scalar, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("shape", [(8, 12, 16)])
@pytest.mark.parametrize("dim", [-1])  # currently only last dim is supported by tpc
@pytest.mark.parametrize("is_scale", [True, False])
def test_softmax_fp8(shape, dim, is_scale):
    input = torch.rand(shape, dtype=torch.bfloat16) * 5.0
    input_hpu = input.to("hpu")

    fn = torch.ops.hpu.softmax_fp8

    if is_scale:
        scale_input = torch.tensor(0.8)
        scale_input_hpu = scale_input.to("hpu")
        scale_output = torch.tensor(1 / 0.8)
        scale_output_hpu = scale_output.to("hpu")
        input = input * scale_input
    else:
        scale_input_hpu = None
        scale_output_hpu = None

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    result = fn(input_hpu, dim, scale_input_hpu, scale_output_hpu)

    result_ref = torch.softmax(input, dim)
    if is_scale:
        result_ref = (result_ref * scale_output).to(torch.float8_e4m3fn)
        assert result.dtype == torch.float8_e4m3fn
    else:
        assert result.dtype == torch.bfloat16

    compare_tensors(result, result_ref, atol=1e-3, rtol=0.2)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("softmax_fp8")
