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


import pytest
import torch
import torch.ao.quantization.fx._decomposed
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    format_tc,
    is_gaudi1,
    is_pytest_mode_compile,
)

pytestmark = [
    pytest.mark.skipif(is_gaudi1(), reason="Not supported by Gaudi"),
]

quant_dtypes = [torch.int8, torch.int32, torch.float8_e5m2, torch.float8_e4m3fn]


def round_if_integer(input, out_dtype):
    if out_dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        return input
    return torch.round(input)


@pytest.mark.parametrize("is_scale_tensor, is_quant_tensor", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("out_dtype", quant_dtypes, ids=format_tc)
def test_quantize_per_tensor(is_scale_tensor, is_quant_tensor, dtype, out_dtype):
    def fn(input, scale, zero_point, quant_min, quant_max, out_dtype):
        return torch.ops.quantized_decomposed.quantize_per_tensor(
            input, scale, zero_point, quant_min, quant_max, out_dtype
        )

    def fn_ref(input, scale, zero_point, quant_min, quant_max, out_dtype):
        return torch.clamp(round_if_integer(input / scale, out_dtype) + zero_point, quant_min, quant_max).to(out_dtype)

    shape = (24, 48)
    scale = 5.0
    zero_point = 10

    input = (torch.rand(shape) * 1000.0).to(dtype)
    input_hpu = input.to("hpu")
    quant_min = 20
    quant_max = 125

    if is_scale_tensor:
        scale = torch.tensor(scale).to(dtype)
        zero_point = torch.tensor(zero_point)
        scale_hpu = scale.to("hpu")
        scale = scale.to(dtype)
        zero_point_hpu = zero_point.to("hpu")
    else:
        scale_hpu = scale
        zero_point_hpu = zero_point

    if is_quant_tensor:
        quant_min = torch.tensor(quant_min)
        quant_max = torch.tensor(quant_max)
        quant_min_hpu = quant_min.to("hpu")
        quant_max_hpu = quant_max.to("hpu")
    else:
        quant_min_hpu = quant_min
        quant_max_hpu = quant_max

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(
            fn,
            backend="hpu_backend",
        )

    result_hpu = fn(input_hpu, scale_hpu, zero_point_hpu, quant_min_hpu, quant_max_hpu, out_dtype)
    result_cpu = fn_ref(input, scale, zero_point, quant_min, quant_max, out_dtype)

    rtol = 0.25 if dtype == torch.bfloat16 else 0.0
    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("quantize_per_tensor")


@pytest.mark.parametrize("is_scale_tensor", [True, False])
@pytest.mark.parametrize("dtype", quant_dtypes, ids=format_tc)
@pytest.mark.parametrize("orig_dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_dequantize_per_tensor(is_scale_tensor, dtype, orig_dtype):
    def fn(input, scale, zero_point, out_dtype):
        return torch.ops.quantized_decomposed.dequantize_per_tensor(
            input, scale, zero_point, 0, 0, input.dtype, out_dtype=out_dtype
        )

    def fn_ref(input, scale, zero_point):
        return (input.to(torch.float) - zero_point) * scale

    shape = (16, 64)
    scale = 0.05
    zero_point = -10

    input = (torch.rand(shape) * 200.0).to(dtype)
    input_hpu = input.to("hpu")

    if is_scale_tensor:
        scale = torch.tensor(scale).to(orig_dtype)
        zero_point = torch.tensor(zero_point)
        scale_hpu = scale.to("hpu")
        zero_point_hpu = zero_point.to("hpu")
    else:
        scale_hpu = scale
        zero_point_hpu = zero_point

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(
            fn,
            backend="hpu_backend",
        )

    result_hpu = fn(input_hpu, scale_hpu, zero_point_hpu, orig_dtype)
    result_cpu = fn_ref(input, scale, zero_point)

    rtol = 0.01 if orig_dtype == torch.bfloat16 else 0.0

    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("dequantize_per_tensor")


def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list


@pytest.mark.parametrize("axis", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("out_dtype", quant_dtypes, ids=format_tc)
def test_quantize_per_channel(axis, dtype, out_dtype):
    def fn(input, scales, zero_points, axis, quant_min, quant_max, out_dtype):
        return torch.ops.quantized_decomposed.quantize_per_channel(
            input, scales, zero_points, axis, quant_min, quant_max, out_dtype
        )

    def fn_ref(input, scales, zero_points, axis, quant_min, quant_max, out_dtype):
        input, permute_axis_list = _permute_to_axis_zero(input, axis)
        res = torch.zeros_like(input)

        for i in range(input.size(0)):
            res[i] = torch.clamp(
                round_if_integer(input[i] * (1.0 / scales[i]), out_dtype) + zero_points[i],
                quant_min,
                quant_max,
            )

        out = res.permute(tuple(permute_axis_list))
        return out.to(out_dtype)

    shape = (8, 16, 12, 20)

    input = (torch.rand(shape) * 500.0).to(dtype)
    input_hpu = input.to("hpu")
    quant_min = 10
    quant_max = 125

    scales = torch.rand((shape[axis],)) * 20.0 + 5.0
    zero_points = torch.randint(0, 10, (shape[axis],))
    scales_hpu = scales.to("hpu")
    scales = scales.to(dtype)
    zero_points_hpu = zero_points.to("hpu")
    zero_points = zero_points.to(dtype)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(
            fn,
            backend="hpu_backend",
        )

    result_hpu = fn(input_hpu, scales_hpu, zero_points_hpu, axis, quant_min, quant_max, out_dtype)
    result_cpu = fn_ref(input, scales, zero_points, axis, quant_min, quant_max, out_dtype)

    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=0.0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("quantize_per_channel")


@pytest.mark.parametrize("axis", [0, -2])
@pytest.mark.parametrize("dtype", quant_dtypes, ids=format_tc)
@pytest.mark.parametrize("orig_dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_dequantize_per_channel(axis, dtype, orig_dtype):
    def fn(input, scales, zero_points, axis, orig_dtype):
        return torch.ops.quantized_decomposed.dequantize_per_channel(
            input, scales, zero_points, axis, 0, 0, input.dtype, out_dtype=orig_dtype
        )

    def fn_ref(input, scales, zero_points, axis, orig_dtype):
        input, permute_axis_list = _permute_to_axis_zero(input, axis)
        res = torch.zeros_like(input, dtype=orig_dtype)

        for i in range(input.size(0)):
            res[i] = (input[i].to(orig_dtype) - zero_points[i]) * scales[i]

        out = res.permute(tuple(permute_axis_list))
        return out

    shape = (4, 22, 16, 8)

    input = (torch.rand(shape) * 200.0).to(dtype)
    input_hpu = input.to("hpu")

    scales = torch.rand((shape[axis],))
    zero_points = torch.randint(-10, 10, (shape[axis],))
    scales_hpu = scales.to("hpu")
    zero_points_hpu = zero_points.to("hpu")

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(
            fn,
            backend="hpu_backend",
        )

    result_hpu = fn(input_hpu, scales_hpu, zero_points_hpu, axis, orig_dtype)
    result_cpu = fn_ref(input, scales, zero_points, axis, orig_dtype)

    rtol = 1e-2 if orig_dtype == torch.bfloat16 else 1e-5

    compare_tensors(result_hpu, result_cpu, atol=0.0, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("dequantize_per_channel")
