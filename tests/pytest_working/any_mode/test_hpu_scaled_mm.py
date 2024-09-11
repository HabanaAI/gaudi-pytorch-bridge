###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    is_gaudi1,
    is_gaudi3,
    is_pytest_mode_compile,
)

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support fp8")]


def get_scale(is_scale, val):
    if is_scale:
        return torch.tensor(val), torch.tensor(val).to("hpu")
    else:
        return torch.tensor(1.0), None


@pytest.mark.parametrize("scale_a", [True, False])
@pytest.mark.parametrize("scale_b", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("scale_result", [True, False])
@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn, "mixed"])
@pytest.mark.parametrize("out_dtype", [torch.float, torch.bfloat16, "fp8"])
def test_hpu_scaled_mm(scale_a, scale_b, bias, scale_result, dtype, out_dtype):
    if scale_result and out_dtype != "fp8":
        pytest.skip("scale_result is applicable only for fp8 output")
    if dtype == "mixed" and not is_gaudi3():
        pytest.skip("mixed fp8 dtypes are supported only on gaudi3")
    if out_dtype == "fp8" and bias:
        pytest.skip("bias is not supported for fp8 output")
    if out_dtype == "fp8" and not (scale_a and scale_b and scale_result):
        pytest.skip("fp8 output is applicable only for all scales given")

    shape_a = (12, 16)
    shape_b = (16, 24)
    fn = torch._scaled_mm

    if dtype == "mixed":
        dtype_a = torch.float8_e5m2
        dtype_b = torch.float8_e4m3fn
    else:
        dtype_a = dtype
        dtype_b = dtype

    if out_dtype == "fp8":
        out_dtype = dtype_a
        ref_dtype = torch.bfloat16
    else:
        ref_dtype = out_dtype

    a = (torch.rand(shape_a) * 5.0).to(dtype_a)
    ah = a.to("hpu")
    a = a.to(ref_dtype)

    b = (torch.rand(shape_b) * 10.0).to(dtype_b)
    bh = b.to("hpu")
    b = b.to(ref_dtype)

    scale_a_cpu, scale_a_hpu = get_scale(scale_a, 2.71)
    scale_b_cpu, scale_b_hpu = get_scale(scale_b, 3.14)
    scale_result_cpu, scale_result_hpu = get_scale(scale_result, 0.05)

    if bias:
        bias_cpu = (torch.randn(shape_b[1]) * 50.0).to(ref_dtype)
        bias_hpu = bias_cpu.to("hpu")
    else:
        bias_cpu = 0.0
        bias_hpu = None

    fn = compile_function_if_compile_mode(fn)

    res_hpu, amax_hpu = fn(
        ah,
        bh,
        bias=bias_hpu,
        out_dtype=out_dtype,
        scale_a=scale_a_hpu,
        scale_b=scale_b_hpu,
        scale_result=scale_result_hpu,
    )
    res_cpu = torch.mm(a, b) * (scale_a_cpu * scale_b_cpu).to(ref_dtype) * scale_result_cpu.to(ref_dtype) + bias_cpu

    if out_dtype == torch.float:
        rtol = 1e-2
    elif out_dtype == torch.bfloat16:
        rtol = 0.015
    elif out_dtype == torch.float8_e5m2:
        rtol = 0.25
    elif out_dtype == torch.float8_e4m3fn:
        rtol = 0.125

    if ref_dtype != out_dtype:
        res_cpu = res_cpu.to(out_dtype).to(ref_dtype)
        amax_cpu = res_cpu.float().abs().max()
        compare_tensors(amax_hpu, amax_cpu, atol=1e-3, rtol=rtol)

    compare_tensors(res_hpu, res_cpu, atol=1e-3, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_scaled_mm")
