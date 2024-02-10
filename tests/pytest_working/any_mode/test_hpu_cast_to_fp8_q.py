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
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    is_gaudi1,
    is_pytest_mode_compile,
)

pytestmark = pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support fp8")

MAX_VAL = {torch.float8_e5m2: 57344.0, torch.float8_e4m3fn: 240.0}

DEFAULT_BIAS = {torch.float8_e5m2: 15, torch.float8_e4m3fn: 7}

FP8_143_BIASES = [3, 7, 11, 15]

@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
@pytest.mark.parametrize("exp_bias", FP8_143_BIASES)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("cut_graph", [True, False])
def test_cast_to_fp8_exp_bias(exp_bias, dtype, fp8_dtype, cut_graph):
    shape_a = (6, 8)
    shape_b = (8, 12)

    # exp_bias increases or decreases the dynamic range of fp8 dtype.
    # e.g. the default range of fp8_143 is +/-240.0, but with exp_bias=3
    # it is 240.0*2^4 = 3840.
    # Inputs are scaled to ensure that values outside the default range
    # pass the test.
    input_factor = MAX_VAL[fp8_dtype] * pow(2, DEFAULT_BIAS[fp8_dtype] - exp_bias)
    a = (torch.rand(shape_a, dtype=dtype) * input_factor).to("hpu")
    b = (torch.rand(shape_b, dtype=dtype) * input_factor).to("hpu")

    if cut_graph:

        def fn1(a, b):
            a_scaled = torch.ops.hpu.cast_to_fp8_q(a, fp8_dtype, exp_bias=exp_bias)
            b_scaled = torch.ops.hpu.cast_to_fp8_q(b, fp8_dtype, exp_bias=exp_bias)

            return (a_scaled, b_scaled)

        def fn2(a, b, a_scaled, b_scaled):
            result_scaled = torch.ops.hpu.fp8_gemm_v2(
                a_scaled, False, b_scaled, False, None, dtype, None, None, None, False
            )
            result_ref = torch.matmul(a, b)
            return result_scaled, result_ref

        if is_pytest_mode_compile():
            clear_t_compile_logs()
            torch._dynamo.reset()
            fn1 = torch.compile(fn1, backend="hpu_backend")
            fn2 = torch.compile(fn2, backend="hpu_backend")

        a_s, b_s = fn1(a, b)
        a_s.cpu()
        result_scaled, result_ref = fn2(a, b, a_s, b_s)
    else:

        def fn(a, b):
            a_scaled = torch.ops.hpu.cast_to_fp8_q(a, fp8_dtype, exp_bias=exp_bias)
            b_scaled = torch.ops.hpu.cast_to_fp8_q(b, fp8_dtype, exp_bias=exp_bias)

            result_scaled = torch.ops.hpu.fp8_gemm_v2(
                a_scaled, False, b_scaled, False, None, dtype, None, None, None, False
            )
            result_ref = torch.matmul(a, b)
            return result_scaled, result_ref

        if is_pytest_mode_compile():
            clear_t_compile_logs()
            torch._dynamo.reset()
            fn = torch.compile(fn, backend="hpu_backend")

        result_scaled, result_ref = fn(a, b)

    assert torch.allclose(result_scaled.cpu(), result_ref.cpu(), rtol=0.1, atol=0.1)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("cast_to_fp8_q")


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
def test_t():
    input = torch.randn((100, 200)) * 800.0
    input_hpu = input.to("hpu")

    def fn(input):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        transposed = input_scaled.t()
        return transposed.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(input_hpu)
    assert torch.allclose(res.cpu(), input.t(), rtol=0.125, atol=0.0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
def test_transpose():
    input = torch.randn((5, 10, 15, 20)) * 800.0
    input_hpu = input.to("hpu")

    def fn(input):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        transposed = input_scaled.transpose(1, 3)
        return transposed.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(input_hpu)
    assert torch.allclose(res.cpu(), input.transpose(1, 3), rtol=0.125, atol=0.0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
def test_permute():
    input = torch.randn((5, 10, 15, 20)) * 800.0
    input_hpu = input.to("hpu")

    def fn(input):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        permuted = input_scaled.permute((2, 3, 1, 0))
        return permuted.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(input_hpu)
    assert torch.allclose(res.cpu(), input.permute((2, 3, 1, 0)), rtol=0.125, atol=0.0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
def test_reshape():
    input = torch.randn((100, 200)) * 800.0
    input_hpu = input.to("hpu")

    def fn(input):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        reshaped = input_scaled.reshape((2, 50, 10, 20))
        return reshaped.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(input_hpu)
    assert torch.allclose(
        res.cpu(), input.reshape((2, 50, 10, 20)), rtol=0.125, atol=0.0
    )


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
def test_expand():
    input = torch.randn((10, 1, 20)) * 800.0
    input_hpu = input.to("hpu")

    def fn(input):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        expanded = input_scaled.expand((10, 4, 20))
        return expanded.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(input_hpu)
    assert torch.allclose(res.cpu(), input.expand((10, 4, 20)), rtol=0.125, atol=0.0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
@pytest.mark.parametrize("axis", [(1, 2, 3), 1])
def test_squeeze(axis):
    input = torch.randn((10, 1, 20, 1, 5)) * 800.0
    input_hpu = input.to("hpu")

    def fn(input):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        squeezed = input_scaled.squeeze(axis)
        return squeezed.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(input_hpu)
    assert torch.allclose(res.cpu(), input.squeeze(axis), rtol=0.125, atol=0.0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Not supported in t.compile yet")
def test_select():
    input = torch.randn((10, 20, 30)) * 800.0
    input_hpu = input.to("hpu")

    def fn(input):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        res1 = input_scaled[3]
        res2 = input_scaled[:, 4]
        res3 = input_scaled[:, :, 5]
        return res1.float(), res2.float(), res3.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res1, res2, res3 = fn(input_hpu)
    res1_ref = input[3]
    res2_ref = input[:, 4]
    res3_ref = input[:, :, 5]
    assert torch.allclose(res1.cpu(), res1_ref, rtol=0.125, atol=0.0)
    assert torch.allclose(res2.cpu(), res2_ref, rtol=0.125, atol=0.0)
    assert torch.allclose(res3.cpu(), res3_ref, rtol=0.125, atol=0.0)

@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
def test_copy():
    input = torch.randn((100, 200)) * 800.0
    output = torch.zeros((100, 200))
    input_hpu = input.to("hpu")
    output_hpu = output.to("hpu")

    def fn(input, output):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        output_scaled = torch.ops.hpu.cast_to_fp8_q(
            output, torch.float8_e4m3fn, exp_bias=3
        )
        output_scaled.copy_(input_scaled)
        return output_scaled.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(input_hpu, output_hpu)
    assert torch.allclose(res.cpu(), input, rtol=0.125, atol=0.0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
def test_copy_cut():
    input = torch.randn((100, 200)) * 800.0
    output = torch.zeros((100, 200))
    input_hpu = input.to("hpu")
    output_hpu = output.to("hpu")

    def fn1(input, output):
        input_scaled = torch.ops.hpu.cast_to_fp8_q(
            input, torch.float8_e4m3fn, exp_bias=3
        )
        output_scaled = torch.ops.hpu.cast_to_fp8_q(
            output, torch.float8_e4m3fn, exp_bias=3
        )
        return input_scaled, output_scaled

    def fn2(input, output):
        output.copy_(input)

    def fn3(output):
        return output.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn1 = torch.compile(fn1, backend="hpu_backend")
        fn2 = torch.compile(fn2, backend="hpu_backend")
        fn3 = torch.compile(fn3, backend="hpu_backend")

    input_scaled, output_scaled = fn1(input_hpu, output_hpu)
    input_scaled.cpu()
    fn2(input_scaled, output_scaled)
    output_scaled.cpu()
    res = fn3(output_scaled)
    assert torch.allclose(res.cpu(), input, rtol=0.125, atol=0.0)


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-174552")
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Not supported in t.compile yet")
def test_index_copy():
    a = torch.zeros(5, 3)
    a_hpu = a.to("hpu")
    b = torch.rand((3, 3)) * 1000.0
    b_hpu = b.to("hpu")
    index = torch.tensor([0, 4, 2])
    index_hpu = index.to("hpu")

    def fn(a, b, index):
        a8 = torch.ops.hpu.cast_to_fp8_q(a, torch.float8_e4m3fn, exp_bias=3)
        b8 = torch.ops.hpu.cast_to_fp8_q(b, torch.float8_e4m3fn, exp_bias=3)
        a8.index_copy_(0, index, b8)
        return a8.float()

    if is_pytest_mode_compile():
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    res = fn(a_hpu, b_hpu, index_hpu)
    res.cpu()

    assert torch.allclose(res.cpu(), a.index_copy_(0, index, b), rtol=0.125, atol=0.0)
