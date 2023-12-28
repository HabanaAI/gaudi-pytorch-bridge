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
from enum import Enum
from fp8_utils import (
    simulateFp8Precision,
    FP8_MAX,
)
from test_utils import (
    clear_t_compile_logs,
    check_ops_executed_in_jit_ir,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
    format_tc,
)

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht

ht.disable_dynamic_shape()

pytestmark = [
    pytest.mark.skipif(is_gaudi1(), reason="Gaudi1 doesn't support fp8"),
]

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
def test_cast_to_fp8_v2(
    shape, dtype, stochastic, is_amax, scale_mode, axis, out_dtype
):
    hpu = torch.device("hpu")
    input_pos = torch.rand(shape, dtype=dtype) * 30 + 10
    input_neg = -input_pos
    input = torch.cat((input_pos, input_neg))

    scale_shape = None
    if not scale_mode:
        scale = torch.tensor(1.0)
        scale_hpu = None
    elif scale_mode in [ScaleMode.TENSOR, ScaleMode.SCALAR]:
        scale_val = 1.3
        scale = torch.tensor(scale_val)
        scale_hpu = (
            scale.to("hpu") if scale_mode == ScaleMode.TENSOR else scale_val
        )
    elif scale_mode in [ScaleMode.TENSOR_CHANNEL, ScaleMode.SCALAR_CHANNEL]:
        scale_arr = (
            (np.random.rand(input.shape[-1 - axis]) * 2.0 + 0.5).astype(
                np.float32
            )
        ).tolist()
        scale = torch.tensor(scale_arr)
        if axis > 0:
            scale = torch.unsqueeze(scale, -1)
            if scale_mode == ScaleMode.SCALAR_CHANNEL:
                scale_shape = scale.shape
        scale_hpu = (
            scale.to("hpu")
            if scale_mode == ScaleMode.TENSOR_CHANNEL
            else scale_arr
        )
    scale_inv = scale.reciprocal()
    scaled_input_low_precision = simulateFp8Precision(
        input * scale.to(dtype), out_dtype
    )
    unscaled_input = scaled_input_low_precision * scale_inv.to(dtype)

    scale_inv_hpu = scale_inv.to(hpu) if scale_mode else None

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
        uncasted = torch.ops.hpu.cast_from_fp8(casted, scale_inv, dtype)
        return casted, amax, uncasted

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

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

    if stochastic:
        assert torch.allclose(
            uncasted.cpu(), unscaled_input, rtol=0.26, atol=0.0
        )
    else:
        assert torch.equal(uncasted.cpu(), unscaled_input)

    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
    else:
        assert amax.numel() == 0

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2", "cast_from_fp8"})


@pytest.mark.parametrize("shape", [(16, 24, 8), (64, 48)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("is_scale_152", [True, False])
@pytest.mark.parametrize("is_scale_143", [True, False])
def test_cast_to_fp8_hybrid(
    shape, dtype, stochastic, is_amax, is_scale_152, is_scale_143
):
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

    scaled_input_low_precision_152 = simulateFp8Precision(
        input * scale_152.to(dtype), torch.float8_e5m2
    )
    unscaled_input_152 = scaled_input_low_precision_152 * scale_152_inv.to(
        dtype
    )

    scaled_input_low_precision_143 = simulateFp8Precision(
        input * scale_143.to(dtype), torch.float8_e4m3fn
    )
    unscaled_input_143 = scaled_input_low_precision_143 * scale_143_inv.to(
        dtype
    )

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
        uncasted_152 = torch.ops.hpu.cast_from_fp8(
            casted_152, scale_152_inv, dtype
        )
        uncasted_143 = torch.ops.hpu.cast_from_fp8(
            casted_143, scale_143_inv, dtype
        )

        return casted_152, casted_143, amax, uncasted_152, uncasted_143

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

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
        assert torch.allclose(
            uncasted_152.cpu(), unscaled_input_152, rtol=0.26, atol=0.0
        )
        assert torch.allclose(
            uncasted_143.cpu(), unscaled_input_143, rtol=0.26, atol=0.0
        )
    else:
        rtol = 0.01 if dtype == torch.bfloat16 else 0.0
        assert torch.allclose(
            uncasted_152.cpu(), unscaled_input_152, rtol=rtol, atol=0.0
        )
        assert torch.allclose(
            uncasted_143.cpu(), unscaled_input_143, rtol=rtol, atol=0.0
        )

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
def test_fp8_gemm_v2(
    shapeA, shapeB, bias, accumulate, scaleA, scaleB, dtype, fp8_dtype
):
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
        A8, _ = torch.ops.hpu.cast_to_fp8_v2(
            A_hpu, scaleA, False, False, fp8_dtype
        )
        B8, _ = torch.ops.hpu.cast_to_fp8_v2(
            B_hpu, scaleB, False, False, fp8_dtype
        )
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

    if bias:
        result_ref = result_ref + bias_tensor
    if accumulate:
        result_ref = result_ref + out
    result = result.cpu()

    percentage_diff = torch.abs(
        (((result - result_ref) / result_ref) * 100).to(torch.int)
    )
    assert np.amax(percentage_diff.numpy()) <= 15

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2", "fp8_gemm_v2"})


@pytest.mark.parametrize("scaleA", [16, 14])
@pytest.mark.parametrize("scaleB", [0.0625, 0.23])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_gemm_v2_scalar_optimization(scaleA, scaleB, dtype):
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
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

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


# For manual testing with below flags
# ENABLE_EXPERIMENTAL_FLAGS=true FUSE_CONVERT_TO_MME=1 PT_HPU_INFERENCE_MODE=1
# Post synapse graphs should contain raw GEMM node with exp_bias set on inputs and output.
@pytest.mark.parametrize("scale_a", [16.0, 1.0, 0.0625, 0.00390625])
@pytest.mark.parametrize("scale_b", [16.0, 1.0, 0.0625, 0.00390625])
@pytest.mark.parametrize("scale_out", [16.0, 1.0, 0.0625, 256.0])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def DISABLED_test_fp8_gemm_v2_bias_optimization(
    scale_a, scale_b, scale_out, dtype
):
    a = (torch.rand(4, 8) * 5).to(torch.float8_e4m3fn).to("hpu")
    b = (torch.rand(8, 12) * 5).to(torch.float8_e4m3fn).to("hpu")

    scale_a_t = torch.tensor(scale_a).to("hpu")
    scale_b_t = torch.tensor(scale_b).to("hpu")
    scale_out_t = torch.tensor(scale_out).to("hpu")

    res_fp8_scalar, _ = torch.ops.hpu.cast_to_fp8_v2(
        torch.ops.hpu.fp8_gemm_v2(
            a, False, b, False, None, dtype, scale_a, scale_b, None, False
        ),
        scale_out,
        False,
        False,
        torch.float8_e4m3fn,
    )

    res_scalar_cpu = res_fp8_scalar.cpu().float()

    res_fp8_tensor, _ = torch.ops.hpu.cast_to_fp8_v2(
        torch.ops.hpu.fp8_gemm_v2(
            a, False, b, False, None, dtype, scale_a_t, scale_b_t, None, False
        ),
        scale_out_t,
        False,
        False,
        torch.float8_e4m3fn,
    )

    rtol = 1e-3 if dtype == torch.float else 1e-2
    compare_tensors(res_fp8_tensor, res_scalar_cpu, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("shape", [(8, 2, 2, 5)])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + fp8_dtypes)
def test_in_place_interleave(shape, dtype):
    input = torch.randn(shape, dtype=torch.bfloat16) * 10.0
    input_hpu = input.to("hpu")
    if dtype != torch.bfloat16:
        input_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(
            input_hpu, None, False, False, dtype
        )
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
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

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
@pytest.mark.parametrize("kernel", [(2, 2), (4, 6)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
@pytest.mark.parametrize("padding", [(0, 0), (1, 1)])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes)
def test_conv2d_fp8(
    N,
    C,
    H,
    W,
    out_channels,
    scaleA,
    scaleB,
    kernel,
    stride,
    padding,
    bias,
    out_dtype,
    fp8_dtype,
):
    if pytest.mode == "eager" and kernel == (4, 6):
        pytest.skip("Configuration not supported")

    input_cpu = (
        torch.rand((N, C, H, W), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    )
    input_hpu = input_cpu.to("hpu").to(fp8_dtype)

    weight_cpu = (
        torch.rand((out_channels, C, kernel[0], kernel[1]), dtype=out_dtype)
        .to(fp8_dtype)
        .to(out_dtype)
    )
    weight_hpu = weight_cpu.to("hpu").to(fp8_dtype)

    bias_cpu = (
        torch.rand(out_channels, dtype=out_dtype).to(fp8_dtype).to(out_dtype)
        if bias
        else None
    )
    bias_hpu = bias_cpu.to("hpu") if bias else None

    scaleA_cpu = 1
    scaleB_cpu = 1
    scaleA_hpu = None
    scaleB_hpu = None

    if scaleA:
        scaleA_cpu = torch.tensor(1.4).to(out_dtype)
        scaleA_hpu = scaleA_cpu.to("hpu")
    if scaleB:
        scaleB_cpu = torch.tensor(2.3).to(out_dtype)
        scaleB_hpu = scaleB_cpu.to("hpu")

    fn = torch.ops.hpu.conv2d_fp8

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="aot_hpu_training_backend")

    conv = fn(
        input_hpu,
        weight_hpu,
        bias_hpu,
        stride,
        padding,
        1,
        1,
        out_dtype,
        scaleA_hpu,
        scaleB_hpu,
    )
    conv_ref = torch.nn.functional.conv2d(
        input_cpu, weight_cpu, bias_cpu, stride, padding, 1, 1
    ) * (scaleA_cpu * scaleB_cpu)

    if out_dtype == torch.bfloat16 and (scaleA or scaleB):
        rtol = 0.02
    else:
        rtol = 1e-2

    compare_tensors(conv, conv_ref, atol=1e-2, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("conv2d_fp8")
