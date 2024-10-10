# ******************************************************************************
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

import pytest
import torch
import torch.nn.functional as F
from test_utils import evaluate_fwd_bwd_kernel, evaluate_fwd_inplace_kernel, evaluate_fwd_kernel

# N - batch
# H - input height
# W - input width
# C - input channels
mnist_test_cast_list = [
    # N, H, W, C
    (64, 24, 24, 20),
    (64, 7, 7, 50),
]

test_case_list = [
    #  N, H, W, C,
    (8, 24, 24, 3),
] + mnist_test_cast_list

unary_op_list = [
    F.relu,
    torch.tanh,
    torch.nn.functional.gelu,
    torch.norm,
    torch.sigmoid,
    torch.sqrt,
    torch.reciprocal,
    torch.floor,
    torch.round,
    torch.rsqrt,
    torch.log,
    torch.log2,
    torch.sign,
    torch.sgn,
    torch.sin,
    torch.cos,
]

unary_op_0d_list = [
    torch.sin,
    torch.cos,
]

unary_special_op_list = [
    [torch.isfinite, None],
    [torch.isinf, None],
    [torch.isposinf, None],
    [torch.isneginf, None],
    [torch.isnan, None],
    # Disabled due to TypeError: isinf() got an unexpected keyword argument 'out'
    # isinf does not have out variant in python API
    # [torch.isinf, "out"],
    [torch.isposinf, "out"],
    [torch.isneginf, "out"],
]

unary_inplace_op_list = [
    ("relu_"),
    ("tanh_"),
    ("erf_"),
    ("exp_"),
    ("reciprocal_"),
    ("floor_"),
    ("round_"),
    ("rsqrt_"),
    ("log_"),
    ("log2_"),
    ("sign_"),
    ("sgn_"),
    ("abs_"),
]

unary_op_out_list = [
    # op, op params dict
    (torch.tanh, {}),
    (torch.reciprocal, {}),
    (torch.neg, {}),
]

data_type_list = [(torch.float, 0.001)]

full_float_list = data_type_list + [
    (torch.bfloat16, 0.01),
    (torch.float64, 0.001),
]

full_int_list = [
    (torch.int32, 0),
]

full_type_list = full_float_list + full_int_list


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", unary_op_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_unary_op(N, H, W, C, unary_op, dtype, tol):
    kernel_params = {}
    if unary_op == torch.norm:
        kernel_params = {"input": torch.randn(N, C, H, W).to(dtype), "p": 6.0}
    elif unary_op == torch.rsqrt:
        kernel_params = {"input": torch.add(torch.rand(N, C, H, W, requires_grad=True), 1).to(dtype)}
    elif unary_op == torch.log or unary_op == torch.log2:
        kernel_params = {"input": torch.arange(1, 100, 0.1, dtype=dtype, requires_grad=True)}
    else:
        kernel_params = {"input": torch.randn(N, C, H, W, requires_grad=True).to(dtype)}

    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params, atol=tol, rtol=tol)


@pytest.mark.parametrize("unary_op, out", unary_special_op_list)
@pytest.mark.parametrize("dtype, tol", full_type_list)
def test_hpu_special_unary_op(unary_op, out, dtype, tol):
    kernel_params = (
        {"input": torch.tensor([0.0, -0.0, math.inf, -math.inf, math.nan, +1.0, -1.0]).to(dtype)}
        if any(dtype == dt_tuple[0] for dt_tuple in full_float_list)
        else {"input": torch.tensor([0, 1, -1]).to(dtype)}
    )
    if out is not None:
        kernel_params[out] = torch.empty(kernel_params["input"].size(), dtype=torch.bool)
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params, atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize(
    "unary_op",
    [
        pytest.param(
            op,
            marks=pytest.mark.xfail(reason="results mismatch") if op in (torch.sgn, torch.nn.functional.gelu) else [],
        )
        for op in unary_op_list
    ],
)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_unary_op_fwd_bwd(N, H, W, C, unary_op, dtype, tol):
    kernel_params_fwd = {}
    if unary_op == torch.norm:
        kernel_params_fwd = {
            "input": torch.randn(N, C, H, W, requires_grad=True).to(dtype),
            "p": 6.0,
        }
        bwd_tensors = [torch.tensor(1).to(dtype)]
    elif unary_op == torch.rsqrt:
        kernel_params_fwd = {"input": torch.add(torch.rand(N, C, H, W, requires_grad=True), 1).to(dtype)}
        bwd_tensors = [torch.randn(N, C, H, W).to(dtype)]
    elif unary_op == torch.log or unary_op == torch.log2:
        kernel_params_fwd = {"input": torch.arange(1, 100, 0.1, dtype=dtype, requires_grad=True)}
        bwd_tensors = [torch.arange(1, 100, 0.1, dtype=dtype)]
    else:
        # TODO: extend that test to all features
        kernel_params_fwd = {"input": torch.randn(N, C, H, W, requires_grad=True).to(dtype)}
        bwd_tensors = [torch.randn(N, C, H, W).to(dtype)]

    evaluate_fwd_bwd_kernel(
        kernel=unary_op,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_gelu_op_fwd_bwd(N, H, W, C, dtype, tol):
    kernel_params_fwd = {"input": torch.randn(N, C, H, W, requires_grad=True).to(dtype)}
    bwd_tensors = [torch.randn(N, C, H, W).to(dtype)]
    evaluate_fwd_bwd_kernel(
        kernel=torch.nn.functional.gelu,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
        atol=0.003,
        rtol=0.003,
        grad_on_grad_enable=False,
    )


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_inplace_op", unary_inplace_op_list)
def test_hpu_unary_inplace_op(N, H, W, C, unary_inplace_op):
    if unary_inplace_op == "rsqrt_":
        in_out_tensor = torch.add(torch.rand(N, C, H, W), 1)
    elif unary_inplace_op == "log_" or unary_inplace_op == "log2_":
        in_out_tensor = torch.arange(1, 100, 0.1)
    else:
        in_out_tensor = torch.randn(N, C, H, W)

    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name=unary_inplace_op,
        kernel_params=None,
    )


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op, kernel_params_fwd", unary_op_out_list)
def test_hpu_binary_op_out_intype(N, H, W, C, unary_op, kernel_params_fwd):
    kernel_params_fwd["input"] = torch.randn(N, C, H, W)
    kernel_params_fwd["out"] = torch.empty((N, C, H, W))
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("unary_op", unary_op_0d_list)
def test_hpu_binary_op_0D(unary_op):
    kernel_params_fwd = {}
    kernel_params_fwd["input"] = torch.tensor(1.0)
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params_fwd)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("lp_norm_op", [torch.norm])
@pytest.mark.parametrize("value", [11.0, 6.0])
def test_hpu_lp_norm_op_fwd_bwd(N, H, W, C, lp_norm_op, value):
    kernel_params_fwd = {
        "input": torch.randn(N, C, H, W, requires_grad=True, dtype=torch.float),
        "p": value,
    }
    bwd_tensors = [torch.tensor(1, dtype=torch.float)]
    evaluate_fwd_bwd_kernel(
        kernel=lp_norm_op,
        tensor_list_bwd=bwd_tensors,
        kernel_params_fwd=kernel_params_fwd,
    )


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", [torch.erf, torch.exp])
@pytest.mark.parametrize("dtype, tol", data_type_list)
def test_hpu_unary_op_erf(N, H, W, C, unary_op, dtype, tol):
    kernel_params = {"input": torch.randn(N, C, H, W).to(dtype)}
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params, atol=tol, rtol=tol)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
@pytest.mark.parametrize("unary_op", [torch.norm])
def test_hpu_unary_op_frobenius_norm(N, H, W, C, unary_op):
    kernel_params = {"input": torch.randn(N, C, H, W)}
    evaluate_fwd_kernel(kernel=unary_op, kernel_params=kernel_params, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_unary_op_clamp(N, H, W, C):
    kernel_params = {
        "input": torch.randn(N, C, H, W),
        "min": -0.25,
        "max": 0.25,
    }
    evaluate_fwd_kernel(kernel=torch.clamp, kernel_params=kernel_params)
    evaluate_fwd_kernel(kernel=torch.clamp, kernel_params=kernel_params)


@pytest.mark.parametrize("N, H, W, C", test_case_list)
def test_hpu_unary_op_clamp_inplace(N, H, W, C):
    in_out_tensor = torch.randn(N, C, H, W)
    kernel_params = {"min": -0.25, "max": 0.25}
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name="clamp_",
        kernel_params=kernel_params,
    )
    evaluate_fwd_inplace_kernel(
        in_out_tensor=in_out_tensor,
        kernel_name="clamp_",
        kernel_params=kernel_params,
    )
