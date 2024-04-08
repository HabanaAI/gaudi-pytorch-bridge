# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import random

import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend.config import configuration_flags
from test_utils import format_tc, is_gaudi1, is_pytest_mode_compile

self_shapes_pull = [(4, 4), (2, 3, 4), (5,), (2, 2, 2, 2)]
other_shapes_pull = [(1), (2, 1, 1), (5,), (2, 1, 2)]

scalar_list = [2.0, 3, 0.5]
k_list = [1, 5, 9]

dtypes = [torch.float, torch.bfloat16, torch.long, torch.int, torch.short, torch.int8]

if not is_gaudi1():
    dtypes.append(torch.float16)

verbose = False

ops_with_tensor_variant = [torch._foreach_add, torch._foreach_mul, torch._foreach_div]
ops_without_tensor_variant = [
    torch._foreach_sub,
    torch._foreach_maximum,
    torch._foreach_minimum,
    torch._foreach_clamp_min,
    torch._foreach_clamp_max,
]
ops_list = ops_with_tensor_variant + ops_without_tensor_variant

ops_with_tensor_variant_inplace = [torch._foreach_add_, torch._foreach_mul_, torch._foreach_div_]
ops_without_tensor_variant_inplace = [
    torch._foreach_sub_,
    torch._foreach_maximum_,
    torch._foreach_minimum_,
    torch._foreach_clamp_min_,
    torch._foreach_clamp_max_,
]
ops_list_inplace = ops_with_tensor_variant_inplace + ops_without_tensor_variant_inplace


def generate_tensor_list(shapes, dtypes):
    self_cpu = [torch.randn(shape).to(dtype) for shape, dtype in zip(shapes, dtypes)]
    self_hpu = [tensor.to("hpu") for tensor in self_cpu]
    return self_cpu, self_hpu


def get_tolerance(op, dtype):
    if (op == torch._foreach_div or op == torch._foreach_div_) and dtype == torch.float16:
        return 1e-2, 2e-3
    else:
        return None, None  # therefore default tolerances will be used


@pytest.mark.parametrize("op", ops_with_tensor_variant)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_dtype", [torch.float32, torch.bfloat16, torch.long, torch.int], ids=format_tc)
def test_foreach_tensor(op, k, other_dtype):
    is_eager_fallback = configuration_flags["use_eager_fallback"]
    configuration_flags["use_eager_fallback"] = True

    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)
    other_cpu = (torch.rand(size=()) * 10).to(other_dtype)
    other_hpu = other_cpu.to("hpu")

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other dtype:", other_dtype)

    results_cpu = op(self_cpu, other_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, other_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)
    configuration_flags["use_eager_fallback"] = is_eager_fallback


@pytest.mark.parametrize("op", ops_list)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_scalar", scalar_list)
def test_foreach_scalar(op, k, other_scalar):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other scalar:", other_scalar)

    results_cpu = op(self_cpu, other_scalar)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, other_scalar)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list)
@pytest.mark.parametrize("k,", k_list)
def test_foreach_list(op, k):
    indexes = [random.randint(0, len(self_shapes_pull) - 1) for _ in range(k)]
    self_shapes = [self_shapes_pull[idx] for idx in indexes]
    other_shapes = [other_shapes_pull[idx] for idx in indexes]
    self_dtypes = random.choices(dtypes, k=k)
    other_dtypes = random.choices(dtypes, k=k)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)
    other_cpu, other_hpu = generate_tensor_list(other_shapes, other_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other shapes:", other_shapes)
        print("Other dtypes:", other_dtypes)

    results_cpu = op(self_cpu, other_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, other_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list)
@pytest.mark.parametrize("k,", k_list)
def test_foreach_scalarlist(op, k):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)
    other_scalars = random.choices(scalar_list, k=k)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other scalars:", other_scalars)

    results_cpu = op(self_cpu, other_scalars)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, other_scalars)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_with_tensor_variant_inplace)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_dtype", [torch.float32, torch.bfloat16, torch.long, torch.int], ids=format_tc)
def test_foreach_tensor_inplace(op, k, other_dtype):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)
    for i in range(len(self_dtypes)):
        if not self_dtypes[i].is_floating_point and op == torch._foreach_div_:
            self_dtypes[i] = torch.float32
        self_dtypes[i] = torch.promote_types(self_dtypes[i], other_dtype)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)
    other_cpu = (torch.rand(size=()) * 10).to(other_dtype)
    other_hpu = other_cpu.to("hpu")

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other dtype:", other_dtype)

    op(self_cpu, other_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, other_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list_inplace)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_scalar", scalar_list)
def test_foreach_scalar_inplace(op, k, other_scalar):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)
    for i in range(len(self_dtypes)):
        if not self_dtypes[i].is_floating_point and op == torch._foreach_div_:
            self_dtypes[i] = torch.float32
        if isinstance(other_scalar, float) and not self_dtypes[i].is_floating_point:
            self_dtypes[i] = torch.promote_types(self_dtypes[i], torch.float32)
    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other scalar:", other_scalar)

    op(self_cpu, other_scalar)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, other_scalar)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list_inplace)
@pytest.mark.parametrize("k,", k_list)
def test_foreach_list_inplace(op, k):
    indexes = [random.randint(0, len(self_shapes_pull) - 1) for _ in range(k)]
    self_shapes = [self_shapes_pull[idx] for idx in indexes]
    other_shapes = [other_shapes_pull[idx] for idx in indexes]
    self_dtypes = random.choices(dtypes, k=k)
    other_dtypes = random.choices(dtypes, k=k)

    for i in range(len(self_dtypes)):
        if not self_dtypes[i].is_floating_point and op == torch._foreach_div_:
            self_dtypes[i] = torch.float32
        self_dtypes[i] = torch.promote_types(self_dtypes[i], other_dtypes[i])

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)
    other_cpu, other_hpu = generate_tensor_list(other_shapes, other_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other shapes:", other_shapes)
        print("Other dtypes:", other_dtypes)

    op(self_cpu, other_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, other_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list_inplace)
@pytest.mark.parametrize("k,", k_list)
def test_foreach_scalarlist_inplace(op, k):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)
    other_scalars = random.choices(scalar_list, k=k)

    for i in range(len(self_dtypes)):
        if not self_dtypes[i].is_floating_point and op == torch._foreach_div_:
            self_dtypes[i] = torch.float32
        if isinstance(other_scalars[i], float) and not self_dtypes[i].is_floating_point:
            self_dtypes[i] = torch.promote_types(self_dtypes[i], torch.float32)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other scalars:", other_scalars)

    op(self_cpu, other_scalars)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, other_scalars)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)
