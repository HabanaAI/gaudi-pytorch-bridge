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
from enum import Enum

import numpy as np
import pytest
import torch
from fp8_utils import FP8_MAX, fp8_dtypes, simulateFp8Precision
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_gaudi1,
    is_gaudi2,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
    is_pytest_mode_lazy,
)

Verbose = False

# Disable dynamic shapes
import habana_frameworks.torch.hpu as ht

ht.disable_dynamic_shape()

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support fp8")]


class ScaleMode(Enum):
    TENSOR = 1
    SCALAR = 2
    TENSOR_CHANNEL = 3
    SCALAR_CHANNEL = 4


def cast_to_fp8_v2_common(shape, dtype, stochastic, is_amax, scale_mode, axis, out_dtype):
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
        # to prevent casts optimization
        casted = casted * 1.0
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
        assert amax.dim() == 0
    else:
        assert amax.numel() == 0

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2", "cast_from_fp8"})


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
@pytest.mark.parametrize("out_dtype", fp8_dtypes, ids=format_tc)
def test_cast_to_fp8_v2(dtype, stochastic, is_amax, out_dtype):
    cast_to_fp8_v2_common((64, 48), dtype, stochastic, is_amax, ScaleMode.TENSOR, None, out_dtype)


@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("scale", [None, 1.0])
def test_cast_to_fp8_v2_from_fp8(dtype, stochastic, scale):
    shape = (64, 48)
    input_pos = torch.rand(shape) * 30 + 10
    input_neg = -input_pos
    input_cpu = torch.cat((input_pos, input_neg)).to(dtype)
    input_hpu = input_cpu.to("hpu")

    def fn(input_hpu, scale, stochastic, dtype):
        args = [input_hpu, scale, stochastic, False, dtype]
        casted_hpu, _ = torch.ops.hpu.cast_to_fp8_v2(*args)

        return casted_hpu

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    casted_hpu = fn(input_hpu, scale, stochastic, dtype)
    casted_cpu = casted_hpu.to("cpu")
    assert torch.equal(input_cpu.to(float), casted_cpu.to(float))

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2"})


@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@pytest.mark.parametrize("dst_dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@pytest.mark.parametrize("is_amax", [False, True])
@pytest.mark.parametrize("scale", [None, 1.0])
def test_cast_to_fp8_v2_from_fp8_exception(dtype, dst_dtype, is_amax, scale):
    input_hpu = torch.rand((16, 24, 8)).to(dtype).to("hpu")
    exception_raised = False
    try:
        casted, _ = torch.ops.hpu.cast_to_fp8_v2(input_hpu, scale, False, is_amax, dst_dtype)
        casted.to(dst_dtype).to("cpu")
    except RuntimeError as e:
        exception_raised = True
        if is_amax:
            assert "CastToFp8V2 must have no amax for float8." in str(e)
        elif dtype != dst_dtype:
            assert (
                f"CastToFp8V2 input and output must have the same dtype for float8, but are {str(dtype).replace('torch.f', 'F')} and {str(dst_dtype).replace('torch.f', 'F')}"
                in str(e)
            )
        else:
            raise RuntimeError(f"unexpected exception {str(e)}")

    if is_amax or dtype != dst_dtype:
        assert exception_raised, "Expected exception not raised"


@pytest.mark.parametrize(
    "scale_mode, axis",
    [
        [ScaleMode.SCALAR, None],
        [ScaleMode.TENSOR_CHANNEL, 0],
        [ScaleMode.TENSOR_CHANNEL, 1],
        [ScaleMode.SCALAR_CHANNEL, 0],
        [ScaleMode.SCALAR_CHANNEL, 1],
    ],
)
@pytest.mark.parametrize("out_dtype", fp8_dtypes, ids=format_tc)
def test_cast_to_fp8_v2_scales(scale_mode, axis, out_dtype):
    cast_to_fp8_v2_common((16, 24, 8), torch.bfloat16, False, True, scale_mode, axis, out_dtype)


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("big_tensor", [True, False])
@pytest.mark.parametrize("out_dtype", fp8_dtypes, ids=format_tc)
def test_cast_to_fp8_v2_out_of_range(dtype, stochastic, big_tensor, out_dtype):
    if out_dtype == torch.float8_e5m2:
        input = torch.tensor([100000, 60000, -60000, -100000], dtype=dtype)
        min = torch.finfo(out_dtype).min
        max = torch.finfo(out_dtype).max
    else:
        input = torch.tensor([1000, 500, -500, -1000], dtype=dtype)
        min = -240.0 if is_gaudi2() else torch.finfo(out_dtype).min
        max = 240.0 if is_gaudi2() else torch.finfo(out_dtype).max
    expected = torch.tensor([max, max, min, min], dtype=torch.float)

    # Check big tensor to verify tpc_fuser behavior
    if big_tensor:
        input = input.expand(800, 4).reshape(80, 40)
        expected = expected.expand(800, 4).reshape(80, 40)

    result, _ = torch.ops.hpu.cast_to_fp8_v2(input.to("hpu"), torch.tensor(1.0).to("hpu"), stochastic, False, out_dtype)
    result = result.cpu().float()

    assert torch.equal(result, expected)


# casting bf16 to f8 uses SFTZ rounding mode, which applies
# stochastic rounding also when rounding number between
# 0.0 and f8 min denormal value.
@pytest.mark.xfail(reason="https://jira.habana-labs.com/browse/SW-188398")
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


def cast_to_fp8_hybrid_common(shape, dtype, stochastic, is_amax, is_scale_152, is_scale_143):
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

    uncasted_143 = uncasted_143.cpu()
    uncasted_152 = uncasted_152.cpu()

    rtol = 0.01 if dtype == torch.bfloat16 else 0.0
    assert torch.allclose(uncasted_143, unscaled_input_143, rtol=rtol, atol=0.0)
    if stochastic:
        assert torch.allclose(uncasted_152, unscaled_input_152, rtol=0.26, atol=0.0)
        assert not torch.equal(uncasted_152, unscaled_input_152)
    else:
        assert torch.allclose(uncasted_152, unscaled_input_152, rtol=rtol, atol=0.0)

    if is_amax:
        assert amax.cpu() == torch.max(input.abs())
        assert amax.dim() == 0
    else:
        assert amax.numel() == 0

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_hybrid", "cast_from_fp8"})


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("is_amax", [True, False])
def test_cast_to_fp8_hybrid(dtype, stochastic, is_amax):
    cast_to_fp8_hybrid_common((64, 48), dtype, stochastic, is_amax, True, True)


@pytest.mark.parametrize("is_scale_152", [True, False])
@pytest.mark.parametrize("is_scale_143", [True, False])
def test_cast_to_fp8_hybrid_scales(is_scale_152, is_scale_143):
    cast_to_fp8_hybrid_common((16, 24, 8), torch.bfloat16, True, True, is_scale_152, is_scale_143)


def fp8_gemm_v2_common(shapeA, shapeB, bias, accumulate, scaleA, scaleB, dtype, fp8_dtype):
    hpu = torch.device("hpu")
    A = torch.rand(shapeA[-1] if isinstance(shapeA, list) else shapeA, dtype=dtype) * 10 + 30.0
    A_hpu = A.to(hpu)
    max_A = torch.max(torch.abs(A)).to(torch.float)

    B = torch.rand(shapeB[-1] if isinstance(shapeB, list) else shapeB, dtype=dtype) * 10 + 30.0
    B_hpu = B.to(hpu)
    max_B = torch.max(torch.abs(B)).to(torch.float)

    scaleA_hpu = None
    scaleB_hpu = None
    scaleAInv = None
    scaleBInv = None

    if scaleA == ScaleMode.TENSOR:
        scaleA_hpu = (FP8_MAX[fp8_dtype] / max_A).to(hpu)
        scaleAInv = torch.reciprocal(scaleA_hpu)
    elif scaleA == ScaleMode.SCALAR:
        scaleA_hpu = (FP8_MAX[fp8_dtype] / max_A).item()
        scaleAInv = 1 / scaleA_hpu
        if not scaleB:
            scaleBInv = 1.0

    if scaleB == ScaleMode.TENSOR:
        scaleB_hpu = (FP8_MAX[fp8_dtype] / max_B).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)
    elif scaleB == ScaleMode.SCALAR:
        scaleB_hpu = (FP8_MAX[fp8_dtype] / max_B).item()
        scaleBInv = 1 / scaleB_hpu
        if not scaleA:
            scaleAInv = 1.0
    elif scaleB == ScaleMode.TENSOR_CHANNEL:
        scaleB_hpu = (FP8_MAX[fp8_dtype] / max_B).expand(shapeB[-1]).to(hpu)
        scaleBInv = torch.reciprocal(scaleB_hpu)
    elif scaleB == ScaleMode.SCALAR_CHANNEL:
        scaleB_h = (FP8_MAX[fp8_dtype] / max_B).expand(shapeB[-1])
        scaleBInv = (1 / scaleB_h).numpy().tolist()
        scaleB_hpu = scaleB_h.numpy().tolist()
        if not scaleA:
            scaleAInv = [1.0]

    As = [A[: s[0], : s[1]] for s in shapeA] if isinstance(shapeA, list) else [A]
    As_hpu = [A_hpu[: s[0], : s[1]] for s in shapeA] if isinstance(shapeA, list) else [A_hpu]
    Bs = [B[: s[0], : s[1]] for s in shapeB] if isinstance(shapeB, list) else [B]
    Bs_hpu = [B_hpu[: s[0], : s[1]] for s in shapeB] if isinstance(shapeB, list) else [B_hpu]
    result_ref = [torch.matmul(a.transpose(-2, -1), b) for a, b in zip(As, Bs)]

    out_shape = [rr.shape for rr in result_ref]
    bias_tensor = [torch.rand(s, dtype=dtype) * 10 + 30.0 for s in out_shape]
    bias_tensor_hpu = [t.to(hpu) if bias else None for t in bias_tensor]

    out = [torch.full(s, 1000.0, dtype=dtype) for s in out_shape]
    out_hpu = [t.to(hpu) for t in out]

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
        fn = torch.compile(fn, backend="hpu_backend", dynamic=len(out) > 1)

    result = [
        fn(
            tA,
            scaleA_hpu,
            tB,
            scaleB_hpu,
            tOut,
            dtype,
            scaleAInv,
            scaleBInv,
            tBias,
            accumulate,
        )
        for tA, tB, tOut, tBias in zip(As_hpu, Bs_hpu, out_hpu, bias_tensor_hpu)
    ]

    if bias:
        result_ref = [rRef + tBias for rRef, tBias in zip(result_ref, bias_tensor)]
    if accumulate:
        result_ref = [rRef + o for rRef, o in zip(result_ref, out)]
    result = [r.cpu() for r in result]

    percentage_diff = [torch.abs((((r - rRef) / rRef) * 100).to(torch.int)) for r, rRef in zip(result, result_ref)]
    for pd in percentage_diff:
        assert np.amax(pd.numpy()) <= 15

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"cast_to_fp8_v2", "fp8_gemm_v2"})


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes, ids=format_tc)
def test_fp8_gemm_v2(bias, accumulate, dtype, fp8_dtype):
    fp8_gemm_v2_common((24, 12), (24, 36), bias, accumulate, ScaleMode.TENSOR, ScaleMode.TENSOR, dtype, fp8_dtype)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes, ids=format_tc)
def test_fp8_gemm_v2_ds(bias, accumulate, dtype, fp8_dtype):
    fp8_gemm_v2_common(
        [(24, 12), (22, 10), (26, 16)],
        [(24, 36), (22, 34), (26, 38)],
        bias,
        accumulate,
        ScaleMode.TENSOR,
        ScaleMode.TENSOR,
        dtype,
        fp8_dtype,
    )


@pytest.mark.parametrize(
    "shapeA, shapeB",
    [
        ((2, 1, 4, 2), (1, 3, 4, 8)),
        ((2, 1, 4, 2), (4, 8)),
        ((4, 2), (3, 4, 8)),
    ],
    ids=format_tc,
)
def test_fp8_gemm_v2_shapes(shapeA, shapeB):
    fp8_gemm_v2_common(shapeA, shapeB, False, False, ScaleMode.TENSOR, ScaleMode.TENSOR, torch.float, torch.float8_e5m2)


scale_modes = [
    (ScaleMode.TENSOR, None),
    (ScaleMode.SCALAR, ScaleMode.SCALAR),
    (ScaleMode.SCALAR, None),
    (None, ScaleMode.TENSOR),
    (None, ScaleMode.SCALAR),
    (None, ScaleMode.TENSOR_CHANNEL),
    (None, ScaleMode.SCALAR_CHANNEL),
    (None, None),
]


@pytest.mark.parametrize("scaleA, scaleB", scale_modes)
def test_fp8_gemm_v2_scales(scaleA, scaleB):
    fp8_gemm_v2_common((2, 1, 4, 2), (1, 3, 4, 8), True, False, scaleA, scaleB, torch.bfloat16, torch.float8_e4m3fn)


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
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_fp8_gemm_v2_1d(shape_a, shape_b, transpose_a, transpose_b):
    hpu = torch.device("hpu")
    dtype = torch.bfloat16

    def generate_input(shape, transpose):
        input = (torch.rand(shape, dtype=dtype) * 10 + 30.0).to(torch.float8_e5m2)
        if transpose:
            if len(shape) == 1:
                pytest.skip("Configuration not supported")
            input_hpu = input.transpose(-2, -1).to(hpu)
        else:
            input_hpu = input.to(hpu)

        return input.to(dtype), input_hpu

    A, A_hpu = generate_input(shape_a, transpose_a)
    B, B_hpu = generate_input(shape_b, transpose_b)

    scaleA = torch.tensor(3.14, dtype=dtype)
    scaleB = torch.tensor(0.75, dtype=dtype)
    scaleA_hpu = scaleA.to("hpu")
    scaleB_hpu = scaleB.to("hpu")

    result_ref = torch.matmul(A, B) * torch.mul(scaleA, scaleB)

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
        None,
        False,
    ).cpu()

    percentage_diff = torch.abs((((result - result_ref) / result_ref) * 100).to(torch.int))
    assert np.amax(percentage_diff.numpy()) <= 15

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("fp8_gemm_v2")


@pytest.mark.parametrize("axis", [0, 1])
def test_fp8_gemm_v2_scale_shape(axis):
    shapeA = (12, 24)
    shapeB = (24, 36)
    in_dtype = torch.float8_e4m3fn
    out_dtype = torch.bfloat16

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
        scale_hpu = scale_array

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
@pytest.mark.parametrize("scale_a", [1.0, 0.0625, 0.00390625])
@pytest.mark.parametrize("scale_b", [16.0, 1.0, 0.0625])
@pytest.mark.parametrize("scale_out", [16.0, 0.0625, 256.0])
def test_fp8_gemm_v2_bias_optimization(scale_a, scale_b, scale_out):
    ht.enable_inference_mode()

    a = (torch.rand(4, 8) * 5).to(torch.float8_e4m3fn).to("hpu")
    b = (torch.rand(8, 12) * 5).to(torch.float8_e4m3fn).to("hpu")

    scale_a_t = torch.tensor(scale_a).to("hpu")
    scale_b_t = torch.tensor(scale_b).to("hpu")
    scale_out_t = torch.tensor(scale_out).to("hpu")

    def fn(a, b, scale_a, scale_b, scale_out):
        return torch.ops.hpu.cast_to_fp8_v2(
            torch.ops.hpu.fp8_gemm_v2(a, False, b, False, None, torch.bfloat16, scale_a, scale_b, None, False),
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

    compare_tensors(res_fp8_tensor, res_scalar_cpu, atol=1e-2, rtol=1e-2)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"fp8_gemm_v2", "cast_to_fp8_v2"})
    ht.disable_inference_mode()


@pytest.mark.skipif(not is_pytest_mode_lazy(), reason="Currently supported only in lazy mode.")
@pytest.mark.parametrize("scale_a", [16.0, 1.0, 7.5])
@pytest.mark.parametrize("scale_b", [16.0, 0.00390625, 7.5])
@pytest.mark.parametrize("scale_out", [0.0625, 256.0, 7.5])
def test_fp8_gemm_v2_mark_scales_const(scale_a, scale_b, scale_out):
    ht.enable_inference_mode()
    from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const

    dtype = torch.bfloat16

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
    res_fp8_tensor, _ = fn(a, b, scale_a_t, scale_b_t, scale_out_t)
    res_fp8_tensor.cpu()

    model = TestModel(scale_a_t, scale_b_t, scale_out_t)

    _mark_params_as_const(model)
    _check_params_as_const(model)

    res_fp8_scalar, _ = model(a, b)
    res_scalar_cpu = res_fp8_scalar.cpu().float()

    compare_tensors(res_fp8_tensor, res_scalar_cpu, atol=1e-2, rtol=1e-2)
    ht.disable_inference_mode()


@pytest.mark.parametrize("dtype", [torch.bfloat16] + fp8_dtypes)
def test_in_place_interleave(dtype):
    shape = (8, 2, 2, 5)
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


@pytest.mark.parametrize("scaleA", [True, False])
@pytest.mark.parametrize("scaleB", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("fp8_dtype", fp8_dtypes, ids=format_tc)
@pytest.mark.parametrize("dynamic", [True, False])
def test_conv2d_fp8(scaleA, scaleB, bias, out_dtype, fp8_dtype, dynamic):
    N, C, H, W = (8, 3, 28, 28)
    out_channels = 16
    kernel = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1,)

    input_cpu = torch.rand((N, C, H, W), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    input_hpu = input_cpu.to("hpu").to(fp8_dtype)

    weight_cpu = torch.rand((out_channels, C, kernel[0], kernel[1]), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    weight_hpu = weight_cpu.to("hpu").to(fp8_dtype)

    bias_cpu = torch.rand(out_channels, dtype=out_dtype).to(fp8_dtype).to(out_dtype) if bias else None
    bias_hpu = bias_cpu.to("hpu") if bias else None

    conv_ref_unscaled = torch.nn.functional.conv2d(input_cpu, weight_cpu, None, stride, padding, dilation, 1)

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
        fn = torch.compile(fn, backend="hpu_backend", dynamic=dynamic)

    conv_args = [input_hpu, weight_hpu, bias_hpu, stride, padding, dilation, 1, out_dtype]
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
def test_conv2d_fp8_scalar_optimization(scaleA, scaleB):
    if scaleA == 16 and scaleB == 0.00390625 and is_pytest_mode_eager():
        pytest.skip("Configuration not supported in eager mode yet.")

    out_dtype = torch.float
    fp8_dtype = torch.float8_e4m3fn

    ht.enable_inference_mode()
    N, C, H, W = (4, 3, 12, 12)
    out_channels = 16
    kernel = (2, 2)
    stride = (1,)
    padding = (0, 0)
    dilation = (1, 1)

    input_cpu = torch.rand((N, C, H, W), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    input_hpu = input_cpu.to("hpu").to(fp8_dtype)

    weight_cpu = torch.rand((out_channels, C, kernel[0], kernel[1]), dtype=out_dtype).to(fp8_dtype).to(out_dtype)
    weight_hpu = weight_cpu.to("hpu").to(fp8_dtype)

    fn = torch.ops.hpu.conv2d_fp8

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    conv = fn(input_hpu, weight_hpu, None, stride, padding, dilation, 1, out_dtype, scaleA, scaleB)
    conv_ref = torch.nn.functional.conv2d(input_cpu, weight_cpu, None, stride, padding, dilation, 1) * (scaleA * scaleB)

    compare_tensors(conv, conv_ref, atol=1e-2, rtol=0.02)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("conv2d_fp8")
    ht.disable_inference_mode()


@pytest.mark.skipif(is_pytest_mode_eager(), reason="Not supported in eager mode yet.")
@pytest.mark.parametrize("scale_a", [16.0, 1.0, 7.5])
@pytest.mark.parametrize("scale_b", [16.0, 0.00390625, 7.5])
@pytest.mark.parametrize("scale_out", [0.0625, 256.0, 7.5])
def test_conv2d_fp8_bias_optimization(scale_a, scale_b, scale_out):
    ht.enable_inference_mode()

    N, C, H, W = (4, 3, 12, 12)
    fp8_dtype = torch.float8_e4m3fn
    out_dtype = torch.bfloat16
    out_channels = 16
    kernel = (2, 2)
    stride = (1, 1)
    padding = (0,)
    dilation = (1, 1)

    input_hpu = (torch.rand(N, C, H, W) * 5).to(fp8_dtype).to("hpu")
    weight_hpu = (torch.rand(out_channels, C, kernel[0], kernel[1]) * 5).to(fp8_dtype).to("hpu")

    scale_a_t = torch.tensor(scale_a).to("hpu")
    scale_b_t = torch.tensor(scale_b).to("hpu")
    scale_out_t = torch.tensor(scale_out).to("hpu")

    def fn(input, weight, scale_a, scale_b, scale_out):
        return (
            torch.ops.hpu.cast_to_fp8_v2(
                torch.ops.hpu.conv2d_fp8(
                    input, weight, None, stride, padding, dilation, 1, out_dtype, scale_a, scale_b
                ),
                scale_out,
                False,
                False,
                fp8_dtype,
            )[0]
            + 1.0
        )

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res_fp8_scalar = fn(input_hpu, weight_hpu, scale_a, scale_b, scale_out).cpu().float()
    res_fp8_tensor = fn(input_hpu, weight_hpu, scale_a_t, scale_b_t, scale_out_t).float()

    compare_tensors(res_fp8_tensor, res_fp8_scalar, atol=1e-2, rtol=0.125)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"conv2d_fp8", "cast_to_fp8_v2"})
    ht.disable_inference_mode()


@pytest.mark.parametrize(
    "is_scale, input_dtype",
    [(True, torch.bfloat16), (True, torch.float8_e4m3fn), (False, torch.bfloat16)],
    ids=format_tc,
)
@pytest.mark.parametrize("is_inv_attn_heads", [True, False])
@pytest.mark.parametrize("fused_add_shape", [{}, (5, 4, 4, 6), (5, 1, 1, 6)], ids=format_tc)
def test_softmax_fp8(is_scale, is_inv_attn_heads, fused_add_shape, input_dtype):
    dim = -1  # currently only last dim is supported by tpc
    shape = (5, 4, 4, 6)
    input = torch.rand(shape, dtype=torch.bfloat16) * 4.0

    input_hpu = input.to(input_dtype).to("hpu")
    scale_input = scale_input_hpu = None
    scale_output = scale_output_hpu = None
    inv_attn_heads = inv_attn_heads_hpu = None
    fused_add = fused_add_hpu = None

    if is_scale:
        scale_input = torch.tensor(0.05)
        scale_output = torch.tensor(2.5)
        scale_input_hpu = scale_input.to("hpu")
        scale_output_hpu = scale_output.to("hpu")

    if fused_add_shape and is_scale:  # fused_add supported only for fp8 out
        fused_add = torch.rand(fused_add_shape, dtype=torch.bfloat16)
        fused_add_hpu = fused_add.to("hpu")
    else:
        fused_add_shape = None

    if is_inv_attn_heads:
        inv_attn_heads = torch.tensor(0.1)
        inv_attn_heads_hpu = inv_attn_heads.to("hpu")

    fn = torch.ops.hpu.softmax_fp8
    fn = compile_function_if_compile_mode(fn)

    result = fn(input_hpu, dim, scale_input_hpu, scale_output_hpu, inv_attn_heads_hpu, fused_add_hpu)

    if is_inv_attn_heads:
        input = input * inv_attn_heads
    if is_scale:
        input = input * scale_input
        if fused_add_shape:  # fused_add supported only for fp8 out
            input = input + fused_add

    result_ref_fp32 = torch.softmax(input, dim).to(torch.bfloat16)

    if is_scale:
        result_ref_fp32 = (result_ref_fp32 * scale_output).to(torch.float8_e4m3fn)
        assert result.dtype == torch.float8_e4m3fn
    else:
        assert result.dtype == torch.bfloat16

    compare_tensors(result, result_ref_fp32, atol=1e-2, rtol=0.1251)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("softmax_fp8")


@pytest.mark.parametrize("is_inv_attn_heads", ["tensor", "scalar", None])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_softmax_fp8_scalar(is_inv_attn_heads, input_dtype):
    dim = -1  # currently only last dim is supported by tpc
    shape = (5, 4, 4, 6)

    input = torch.rand(shape, dtype=torch.bfloat16) * 4.0
    input_hpu = input.to(input_dtype).to("hpu")
    scale_input = 0.05
    scale_output = 2.5
    inv_attn_heads = inv_attn_heads_hpu = None

    if is_inv_attn_heads:
        inv_attn_heads = torch.tensor(0.1)
        inv_attn_heads_hpu = inv_attn_heads.to("hpu") if is_inv_attn_heads == "tensor" else 0.1

    fn = torch.ops.hpu.softmax_fp8
    fn = compile_function_if_compile_mode(fn)

    result = fn(input_hpu, dim, scale_input, scale_output, inv_attn_heads_hpu)

    if is_inv_attn_heads:
        input = input * inv_attn_heads
    input = input * scale_input

    result_ref_fp32 = torch.softmax(input, dim).to(torch.bfloat16)

    result_ref_fp32 = (result_ref_fp32 * scale_output).to(torch.float8_e4m3fn)
    assert result.dtype == torch.float8_e4m3fn

    compare_tensors(result, result_ref_fp32, atol=1e-2, rtol=0.1251)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("softmax_fp8")
