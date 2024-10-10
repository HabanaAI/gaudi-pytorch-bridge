# ******************************************************************************
# Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import math
import random

import pytest
import torch
from test_utils import compare_tensors, evaluate_fwd_inplace_kernel, evaluate_fwd_kernel

N = 8
C = 3
H = 24
W = 24

# N - batch
# H - input height
# W - input width
# C - input channels
test_case_list = [
    #  N, H, W, C,
    (
        N,
        H,
        W,
        C,
    ),
]

binary_inplace_op_list = [
    # op, op params dict
    ("add_", {"alpha": 1}),
    ("add_", {"alpha": 0.1}),
    ("sub_", {"alpha": 1}),
    ("sub_", {"alpha": 0.1}),
    ("mul_", {}),
    ("div_", {}),
    ("atan2_", {}),
]

binary_op_list = [
    # op, op params dict
    (torch.add, {}),
    (torch.add, {"alpha": 0.1}),
    (torch.mul, {}),
    (torch.div, {}),
    (torch.max, {}),
    (torch.min, {}),
    (torch.atan2, {}),
]

# This list is used to test tensor_out variants of operators
binary_op_out_list = [
    # op, op params dict
    (torch.div, {}),
    (torch.atan2, {}),
]

data_type_list = [(torch.float, 0.001, {}), (torch.bfloat16, 0.01, {}), (torch.float64, 0.001, {})]


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_out_list)
def test_hpu_binary_op_out_intype(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd["input"] = inT = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(N, C, H, W)
    kernel_params_fwd["out"] = torch.empty((N, C, H, W), dtype=inT.dtype)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_inplace_op_list)
def test_hpu_binary_inplace_op(N, H, W, C, binary_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(N, C, H, W)
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name=binary_op,
        kernel_params=kernel_params_fwd,
    )


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_inplace_op_list)
def test_hpu_binary_inplace_op_broadcast_case1(N, H, W, C, binary_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(H, W)
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name=binary_op,
        kernel_params=kernel_params_fwd,
    )


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_inplace_op_list)
def test_hpu_binary_inplace_op_broadcast_case2(N, H, W, C, binary_op, kernel_params_fwd):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(H, 1)
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name=binary_op,
        kernel_params=kernel_params_fwd,
    )


@pytest.mark.skip
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_inplace_op_pow(N, H, W, C):
    kernel_params_fwd = {}
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params_fwd["exponent"] = torch.randn(N, C, H, W)
    evaluate_fwd_inplace_kernel(in_out_tensor=in_out_tensor, kernel_name="pow_", kernel_params=kernel_params_fwd)


@pytest.mark.skip(reason="IndexError: tuple index out of range")
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_mult_bool_op(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randn(N, C, H, W).to(torch.bool)
    kernel_params_fwd["other"] = torch.randn(N, C, H, W).to(torch.bool)
    evaluate_fwd_kernel(kernel=torch.mul, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("dtype, tol, allowed_ops_for_dtype", data_type_list)
def test_hpu_atan2_special(dtype, tol, allowed_ops_for_dtype):
    if len(allowed_ops_for_dtype) > 0 and torch.atan2 not in allowed_ops_for_dtype:
        pytest.skip()
    binary_op = torch.atan2

    special = [0.0, -0.0, math.inf, -math.inf, math.nan, +1.0, -1.0]
    x = [special] * len(special)
    y = [[v] * len(special) for v in special]

    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor(y).to(dtype)
    kernel_params_fwd["other"] = torch.tensor(x).to(dtype)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd, atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_list)
@pytest.mark.parametrize("dtype, tol, allowed_ops_for_dtype", data_type_list)
def test_hpu_binary_op(N, H, W, C, binary_op, kernel_params_fwd, dtype, tol, allowed_ops_for_dtype):
    if len(allowed_ops_for_dtype) > 0 and binary_op not in allowed_ops_for_dtype:
        pytest.skip()
    kernel_params_fwd["input"] = torch.randn(N, C, H, W).to(dtype)
    kernel_params_fwd["other"] = torch.randn(N, C, H, W).to(dtype)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd, atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_list)
def test_hpu_binary_op_broadcast_case1(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = torch.randn(H, W)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("binary_op, kernel_params_fwd", binary_op_list)
def test_hpu_binary_op_broadcast_case2(N, H, W, C, binary_op, kernel_params_fwd):
    kernel_params_fwd["input"] = torch.randn(N, C, 1, W)
    kernel_params_fwd["other"] = torch.randn(H, 1)
    evaluate_fwd_kernel(kernel=binary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_op_rsub_scalar(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["other"] = round(random.random(), 2)
    kernel_params_fwd["alpha"] = round(random.random(), 2)
    evaluate_fwd_kernel(kernel=torch.rsub, kernel_params=kernel_params_fwd)


@pytest.mark.skip(reason="IndexError: tuple index out of range")
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_op_pow(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["exponent"] = torch.randn(N, C, H, W)
    evaluate_fwd_kernel(kernel=torch.pow, kernel_params=kernel_params_fwd)


@pytest.mark.skip(reason="IndexError: tuple index out of range")
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_op_pow_tensor_scalar(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["exponent"] = random.random()
    evaluate_fwd_kernel(kernel=torch.pow, kernel_params=kernel_params_fwd)


@pytest.mark.skip(reason="IndexError: tuple index out of range")
@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_binary_op_pow_scalar_tensor(N, H, W, C):
    kernel_params_fwd = {}
    kernel_params_fwd["self"] = 3.2
    kernel_params_fwd["exponent"] = torch.randn(N, C, H, W)
    evaluate_fwd_kernel(kernel=torch.pow, kernel_params=kernel_params_fwd)


def test_hpu_binary_op_remainder_tensor_tensor():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor([4, 2]).to(torch.int)
    kernel_params_fwd["other"] = torch.tensor([3, 5]).to(torch.int)
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


def test_hpu_binary_op_remainder_tensor_scalar():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor([4, 2]).to(torch.int)
    kernel_params_fwd["other"] = 3
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


def test_hpu_binary_op_remainder_tensor_tensor_0d():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor(4).to(torch.int)
    kernel_params_fwd["other"] = torch.tensor(3).to(torch.int)
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


def test_hpu_binary_op_remainder_tensor_tensor_inplace():
    input = torch.tensor([4, 8, 2]).to(torch.int)
    other = torch.tensor([3, 5, 7]).to(torch.int)

    input_hpu = input.to("hpu")
    other_hpu = other.to("hpu")

    input.remainder_(other)
    input_hpu.remainder_(other_hpu)

    compare_tensors(input_hpu, input, atol=0.001, rtol=0.001)


def test_hpu_binary_op_remainder_tensor_scalar_inplace():
    input = torch.tensor([4, 8, 2]).to(torch.int)
    other = int(3)

    input_hpu = input.to("hpu")

    input.remainder_(other)
    input_hpu.remainder_(other)
    cpu_out = input_hpu.to("cpu")

    compare_tensors(cpu_out, input, atol=0.001, rtol=0.001)


def test_hpu_binary_op_remainder_tensor_tensor_inplace_0d():
    input = torch.tensor([4, 8, 2]).to(torch.int)
    other = torch.tensor(3).to(torch.int)

    input_hpu = input.to("hpu")
    other_hpu = other.to("hpu")

    input.remainder_(other)
    input_hpu.remainder_(other_hpu)
    cpu_out = input_hpu.to("cpu")

    compare_tensors(cpu_out, input, atol=0.001, rtol=0.001)


def test_hpu_remainder_tensor_op_out_intype():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor([4, 2]).to(torch.int)
    kernel_params_fwd["other"] = torch.tensor([3, 5]).to(torch.int)
    kernel_params_fwd["out"] = torch.tensor([0, 0]).to(torch.int)
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


def test_hpu_remainder_scalar_op_out_intype():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor([4, 2]).to(torch.int)
    kernel_params_fwd["other"] = 3
    kernel_params_fwd["out"] = torch.tensor([0, 0]).to(torch.int)
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


def test_hpu_remainder_tensor_op_out_intype_0d():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor(4).to(torch.int)
    kernel_params_fwd["other"] = 3
    kernel_params_fwd["out"] = torch.tensor(0)
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


def test_hpu_remainder_tensor_op_resizeoutput():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor([4, 2, 7]).to(torch.int)
    kernel_params_fwd["other"] = torch.tensor([3, 5, 11]).to(torch.int)
    kernel_params_fwd["out"] = torch.empty(1).to(torch.int)
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


def test_hpu_remainder_scalar_op_resizeoutput():
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor([4, 2, 7]).to(torch.int)
    kernel_params_fwd["other"] = 3
    kernel_params_fwd["out"] = torch.empty(1).to(torch.int)
    evaluate_fwd_kernel(kernel=torch.remainder, kernel_params=kernel_params_fwd)


if __name__ == "__main__":
    test_hpu_binary_op_broadcast_case2(*test_case_list[0], torch.add, {})
