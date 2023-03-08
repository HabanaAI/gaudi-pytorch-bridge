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
import habana_frameworks.torch.dynamo._custom_op_meta_registrations

def verify_jit(fx_module: torch.fx.GraphModule, op_name: str):
    code = str(fx_module.code)
    f = torch.jit.script(fx_module)
    graph = str(f.graph)
    assert f"torch.ops.hpu.{op_name}" in code
    assert f"hpu::{op_name}" in graph

def verify_not_available(error, op_name):
    assert f"hpu::{op_name} is not available in Eager mode" in str(error.value)

def create_inputs(shape, dtype):
    hpu = torch.device("hpu")
    input = (torch.rand(shape, dtype=dtype)*30 + 10).to(hpu)
    scale = torch.tensor(0.75, dtype=torch.float).to(hpu)
    amax = torch.empty((2, 3), dtype=torch.float).to(hpu)
    amax_temp = torch.tensor(0, dtype=torch.float).to(hpu)
    return input, scale, amax, amax_temp

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_cast_to_fp8(dtype):
    op_name = "cast_to_fp8"
    input_shape = (64, 48)
    input, scale, amax, amax_temp = create_inputs(input_shape, dtype)

    def fn(input, scale, amax, out):
        torch.ops.hpu.cast_to_fp8(input, scale, False, out, amax)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    out = torch.empty(input_shape, dtype=torch.int8, device=input.device)
    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(input, scale, amax_temp, out)
        amax[1][2].copy_(amax_temp)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_cast_transpose(dtype):
    op_name = "fp8_cast_transpose"
    input_shape = (64, 48)
    input, scale, amax, amax_temp = create_inputs(input_shape, dtype)

    def fn(input, scale, amax, out, out_t):
        torch.ops.hpu.fp8_cast_transpose(input, scale, False, out, amax, out_t)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    out = torch.empty(input_shape, dtype=torch.int8, device=input.device)
    out_t = torch.empty((input_shape[1], input_shape[0]), dtype=torch.int8, device=input.device)
    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(input, scale, amax_temp, out, out_t)
        amax[1][2].copy_(amax_temp)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_cast_transpose_bgrad(dtype):
    op_name = "fp8_cast_transpose_bgrad"
    input_shape = (64, 48)
    input, scale, amax, amax_temp = create_inputs(input_shape, dtype)

    def fn(input, scale, amax, out, out_t, bgrad_out):
        torch.ops.hpu.fp8_cast_transpose_bgrad(input, scale, False, out, amax, out_t, bgrad_out)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    out = torch.empty(input_shape, dtype=torch.int8, device=input.device)
    out_t = torch.empty((input_shape[1], input_shape[0]), dtype=torch.int8, device=input.device)
    bgrad_out = torch.empty((input_shape[1],), dtype=dtype, device="hpu")
    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(input, scale, amax_temp, out, out_t, bgrad_out)
        amax[1][2].copy_(amax_temp)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_cast_transpose_bgrad_dgelu(dtype):
    op_name = "fp8_cast_transpose_bgrad_dgelu"
    input_shape = (64, 48)
    input, scale, amax, amax_temp = create_inputs(input_shape, dtype)
    grad = torch.rand(input_shape, dtype=dtype).to("hpu")
    retain = torch.rand(input_shape, dtype=dtype).to("hpu")

    def fn(grad, input, scale, retain, amax, out, out_t, bgrad_out):
        torch.ops.hpu.fp8_cast_transpose_bgrad_dgelu(grad, input, scale, retain, False, out, amax, out_t, bgrad_out)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    out = torch.empty(input_shape, dtype=torch.int8, device=input.device)
    out_t = torch.empty((input_shape[1], input_shape[0]), dtype=torch.int8, device=input.device)
    bgrad_out = torch.empty((input_shape[1],), dtype=dtype, device="hpu")
    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(grad, input, scale, retain, amax_temp, out, out_t, bgrad_out)
        amax[1][2].copy_(amax_temp)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_cast_from_fp8(dtype):
    op_name = "cast_from_fp8"
    input_shape = (64, 48)
    hpu = torch.device("hpu")
    input = torch.randint(low=-127, high=127, size=input_shape, dtype=torch.int8).to(hpu)
    scale = torch.tensor(0.75, dtype=torch.float).to(hpu)

    def fn(input, scale, out_dtype):
        return torch.ops.hpu.cast_from_fp8(input, scale, out_dtype)

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(input, scale, dtype)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_dropout(dtype):
    op_name = "fp8_dropout"
    input_shape = (64, 48)
    input = (torch.rand(input_shape, dtype=dtype)*30 + 10).to("hpu")
    scale = torch.tensor(0.75, dtype=torch.float).to("hpu")

    def fn(input, scale):
        return torch.ops.hpu.fp8_dropout(input, 0.3, scale, False, True)

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    with pytest.raises(RuntimeError) as error:
        result = compiled_fn(input, scale)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_gelu(dtype):
    op_name = "fp8_gelu"
    input_shape = (64, 48)
    input, scale, amax, amax_temp = create_inputs(input_shape, dtype)

    def fn(input, scale, amax, out, retain):
        torch.ops.hpu.fp8_gelu(input, scale, False, out, amax, retain)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    out = torch.empty(input_shape, dtype=torch.int8, device=input.device)
    retain = torch.empty(input_shape, dtype=dtype, device=input.device)
    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(input, scale, amax_temp, out, retain)
        amax[1][2].copy_(amax_temp)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_layernorm(dtype):
    op_name = "fp8_layernorm"
    input_shape = (64, 48)
    input, scale, amax, amax_temp = create_inputs(input_shape, dtype)
    weight = torch.rand((input_shape[1],), dtype=dtype).to("hpu")
    bias = torch.rand((input_shape[1],), dtype=dtype).to("hpu")
    eps = 0.07

    def fn(input, weight, bias, eps, scale, out, amax, mean, istd):
        torch.ops.hpu.fp8_layernorm(input, weight, bias, eps, scale, False, out, amax, mean, istd)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    out = torch.empty(input_shape, dtype=torch.int8, device=input.device)
    mean = torch.empty((input_shape[0],), dtype=torch.float, device="hpu")
    istd = torch.empty((input_shape[0],), dtype=torch.float, device="hpu")
    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(input, weight, bias, eps, scale, out, amax_temp, mean, istd)
        amax[1][2].copy_(amax_temp)
    verify_not_available(error, op_name)

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_fp8_gemm(dtype):
    op_name = "fp8_gemm"
    hpu = torch.device("hpu")
    input_shape_A = (64, 48)
    input_shape_B = (64, 112)
    A = (torch.rand(input_shape_A, dtype=dtype)*10 + 30.0).to(hpu)
    B = (torch.rand(input_shape_B, dtype=dtype)*10 + 30.0).to(hpu)
    scale_A = torch.tensor(0.75, dtype=torch.float).to(hpu)
    scale_B = torch.tensor(1.44, dtype=torch.float).to(hpu)
    out_shape = (input_shape_A[-1],) + (input_shape_B[-1],)
    bias = (torch.rand(out_shape, dtype=dtype)*10 + 30.0).to(hpu)
    out = torch.full(out_shape, 1000.0, dtype=dtype).to(hpu)

    def fn(A, scale_A, B, scale_B, out_dtype, bias, accumulate, out):
        torch.ops.hpu.fp8_gemm(A, scale_A, True, B, scale_B, False, out, out_dtype, bias, accumulate, out)
        return out

    def toy_compiler(fx_module: torch.fx.GraphModule, example_inputs):
        verify_jit(fx_module, op_name)
        return fx_module

    compiled_fn = torch.compile(fn, backend=toy_compiler)

    out = torch.empty(out_shape, dtype=dtype, device=A.device)
    with pytest.raises(RuntimeError) as error:
        casted = compiled_fn(A, scale_A, B, scale_B, dtype, bias, True, out)
    verify_not_available(error, op_name)
