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
from packaging.version import Version, parse
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
    torch._foreach_pow,
]
ops_list = ops_with_tensor_variant + ops_without_tensor_variant

ops_with_tensor_variant_inplace = [torch._foreach_add_, torch._foreach_mul_, torch._foreach_div_]
ops_without_tensor_variant_inplace = [
    torch._foreach_sub_,
    torch._foreach_maximum_,
    torch._foreach_minimum_,
    torch._foreach_clamp_min_,
    torch._foreach_clamp_max_,
    torch._foreach_pow_,
]
ops_list_inplace = ops_with_tensor_variant_inplace + ops_without_tensor_variant_inplace

if Version(parse(torch.__version__).base_version) >= Version("2.4"):
    OP_OUT_DTYPE_NOT_SUPPORTED_ON_HPU = {
        torch._foreach_add: [torch.int8],
        torch._foreach_sub: [torch.int8],
        torch._foreach_pow: [torch.long],
        torch._foreach_add_: [torch.int8],
        torch._foreach_sub_: [torch.int8],
        torch._foreach_pow_: [torch.long],
        torch._foreach_ceil: [torch.long],
        torch._foreach_cosh: [torch.float16],
        torch._foreach_floor: [torch.long],
        torch._foreach_neg: [torch.int8],
        torch._foreach_sinh: [torch.float16],
        torch._foreach_trunc: [torch.long],
        torch._foreach_ceil_: [torch.long],
        torch._foreach_cosh_: [torch.float16],
        torch._foreach_floor_: [torch.long],
        torch._foreach_neg_: [torch.int8],
        torch._foreach_sinh_: [torch.float16],
        torch._foreach_trunc_: [torch.long],
        torch._foreach_addcdiv: [torch.int32, torch.int16, torch.int32, torch.float16, torch.int16, torch.int8],
    }


def generate_tensor_list(shapes, dtypes, non_negative=False):
    self_cpu = [torch.randn(shape).to(dtype) for shape, dtype in zip(shapes, dtypes)]
    if non_negative:
        self_cpu = [torch.abs(tensor) for tensor in self_cpu]
    self_hpu = [tensor.to("hpu") for tensor in self_cpu]
    return self_cpu, self_hpu


def get_tolerance(op, dtype, is_op_compound=False):
    if (op == torch._foreach_div or op == torch._foreach_div_ or is_op_compound) and dtype == torch.float16:
        return 1e-2, 2e-3
    else:
        return None, None  # therefore default tolerances will be used


def _is_python_2_4():
    return Version(parse(torch.__version__).base_version) >= Version("2.4")


def _remove_not_supported_dtypes_on_hpu(op, dtype):
    for not_supported_dtypes in OP_OUT_DTYPE_NOT_SUPPORTED_ON_HPU.get(op, []):
        if not_supported_dtypes in dtype:
            dtype.remove(not_supported_dtypes)


@pytest.mark.parametrize("op", ops_with_tensor_variant)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_dtype", [torch.float32, torch.bfloat16, torch.long, torch.int], ids=format_tc)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_tensor(op, k, other_dtype):
    is_eager_fallback = configuration_flags["use_eager_fallback"]
    configuration_flags["use_eager_fallback"] = True

    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    if _is_python_2_4():
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)

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

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)
    configuration_flags["use_eager_fallback"] = is_eager_fallback


@pytest.mark.parametrize("op", ops_list)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_scalar", scalar_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_scalar(op, k, other_scalar):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    if _is_python_2_4():
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)
        if len(self_dtypes) == 0:
            pytest.skip(reason="Lack of possible types to test for op")

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other scalar:", other_scalar)

    results_cpu = op(self_cpu, other_scalar)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, other_scalar)

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_list(op, k):
    indexes = [random.randint(0, len(self_shapes_pull) - 1) for _ in range(k)]
    self_shapes = [self_shapes_pull[idx] for idx in indexes]
    other_shapes = [other_shapes_pull[idx] for idx in indexes]
    self_dtypes = random.choices(dtypes, k=k)
    other_dtypes = random.choices(dtypes, k=k)

    non_negative = True if op == torch._foreach_pow else False
    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)
    other_cpu, other_hpu = generate_tensor_list(other_shapes, other_dtypes, non_negative=non_negative)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other shapes:", other_shapes)
        print("Other dtypes:", other_dtypes)

    results_cpu = op(self_cpu, other_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, other_hpu)

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_scalarlist(op, k):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    if _is_python_2_4():
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)
        if len(self_dtypes) == 0:
            pytest.skip(reason="Lack of possible types to test for op")

    other_scalars = random.choices(scalar_list, k=len(self_dtypes))
    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other scalars:", other_scalars)

    results_cpu = op(self_cpu, other_scalars)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, other_scalars)

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("self_scalar", scalar_list)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_scalar_and_tensor(self_scalar, k):
    op = torch._foreach_pow

    other_shapes = random.choices(self_shapes_pull, k=k)
    other_dtypes = random.choices(dtypes, k=k)

    other_cpu, other_hpu = generate_tensor_list(other_shapes, other_dtypes)

    if verbose:
        print("Self scalar:", self_scalar)
        print("Other shapes:", other_shapes)
        print("Other dtypes:", other_dtypes)

    results_cpu = op(self_scalar, other_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_scalar, other_hpu)

    for i in range(len(other_dtypes)):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_with_tensor_variant_inplace)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_dtype", [torch.float32, torch.bfloat16, torch.long, torch.int], ids=format_tc)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_tensor_inplace(op, k, other_dtype):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    if _is_python_2_4():
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)

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

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list_inplace)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.parametrize("other_scalar", scalar_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_scalar_inplace(op, k, other_scalar):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    if _is_python_2_4():
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)
        if len(self_dtypes) == 0:
            pytest.skip(reason="Lack of possible types to test for op")

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

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list_inplace)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
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

    non_negative = True if op == torch._foreach_pow_ else False
    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)
    other_cpu, other_hpu = generate_tensor_list(other_shapes, other_dtypes, non_negative=non_negative)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Other shapes:", other_shapes)
        print("Other dtypes:", other_dtypes)

    op(self_cpu, other_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, other_hpu)

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", ops_list_inplace)
@pytest.mark.parametrize("k,", k_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_foreach_scalarlist_inplace(op, k):
    self_shapes = random.choices(self_shapes_pull, k=k)
    self_dtypes = random.choices(dtypes, k=k)

    if _is_python_2_4():
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)
        if len(self_dtypes) == 0:
            pytest.skip(reason="Lack of possible types to test for op")

    other_scalars = random.choices(scalar_list, k=len(self_dtypes))

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

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


compound_foreach_ops = [torch._foreach_addcdiv, torch._foreach_addcmul]
compound_foreach_inplace_ops = [torch._foreach_addcdiv_, torch._foreach_addcmul_]


def create_compound_foreach_tensors(k, cast_to_integer=False, promote_dtype=False, is_value_float=False):
    indexes = [random.randint(0, len(self_shapes_pull) - 1) for _ in range(k)]
    self_shapes = [self_shapes_pull[idx] for idx in indexes]
    tensor_shapes = [other_shapes_pull[idx] for idx in indexes]

    self_dtypes = random.choices(dtypes, k=k)
    tensor1_dtypes = random.choices(dtypes, k=k)
    tensor2_dtypes = random.choices(dtypes, k=k)

    if cast_to_integer:
        self_dtypes = [dtype if dtype.is_floating_point else torch.float32 for dtype in self_dtypes]
        tensor2_dtypes = [dtype if dtype.is_floating_point else torch.float32 for dtype in tensor2_dtypes]

    if promote_dtype:
        for i in range(len(self_dtypes)):
            floating_value = is_value_float[i] if isinstance(is_value_float, list) else is_value_float

            if tensor1_dtypes[i].is_floating_point or tensor2_dtypes[i].is_floating_point or floating_value:
                self_dtypes[i] = torch.promote_types(self_dtypes[i], torch.float32)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)
        print("Tensor shapes:", tensor_shapes)
        print("Tensor1 dtypes:", tensor1_dtypes)
        print("Tensor2 dtypes:", tensor2_dtypes)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)
    tensor1_cpu, tensor1_hpu = generate_tensor_list(tensor_shapes, tensor1_dtypes)
    tensor2_cpu, tensor2_hpu = generate_tensor_list(tensor_shapes, tensor2_dtypes)

    return self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, tensor2_cpu, tensor2_hpu


@pytest.mark.parametrize("op", compound_foreach_ops)
@pytest.mark.parametrize("k", k_list)
@pytest.mark.parametrize("other_dtype", [torch.float32, torch.bfloat16], ids=format_tc)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_compound_foreach_tensor(op, k, other_dtype):
    if _is_python_2_4():
        pytest.skip(reason="aten::<op>.out is not yet supported on HPU")

    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, tensor2_cpu, tensor2_hpu = create_compound_foreach_tensors(
        k, op == torch._foreach_addcdiv
    )

    scalars_cpu = (torch.rand(size=(k,)) * 10).to(other_dtype)
    if _is_python_2_4():
        scalars_hpu = scalars_cpu
    else:
        scalars_hpu = scalars_cpu.to("hpu")

    if verbose:
        print("Scalars tensor:", scalars_cpu)

    results_cpu = op(self_cpu, tensor1_cpu, tensor2_cpu, scalars=scalars_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, tensor1_hpu, tensor2_hpu, scalars=scalars_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype, True)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", compound_foreach_ops)
@pytest.mark.parametrize("k", k_list)
@pytest.mark.parametrize("value", scalar_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_compound_foreach_scalar(op, k, value):
    if _is_python_2_4():
        pytest.skip(reason="aten::<op>.out is not yet supported on HPU")
    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, tensor2_cpu, tensor2_hpu = create_compound_foreach_tensors(
        k, op == torch._foreach_addcdiv
    )

    if verbose:
        print("Value:", value)

    results_cpu = op(self_cpu, tensor1_cpu, tensor2_cpu, value=value)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, tensor1_hpu, tensor2_hpu, value=value)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype, True)
        torch.testing.assert_close(
            results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol, check_dtype=False
        )


@pytest.mark.parametrize("op", compound_foreach_ops)
@pytest.mark.parametrize("k", k_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_compound_foreach_scalarlist(op, k):
    if _is_python_2_4():
        pytest.skip(reason="aten::<op>.out is not yet supported on HPU")
    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, tensor2_cpu, tensor2_hpu = create_compound_foreach_tensors(
        k, op == torch._foreach_addcdiv
    )

    scalars = random.choices(scalar_list, k=k)

    if verbose:
        print("Scalars:", scalars)

    results_cpu = op(self_cpu, tensor1_cpu, tensor2_cpu, scalars=scalars)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, tensor1_hpu, tensor2_hpu, scalars=scalars)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype, True)
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", compound_foreach_inplace_ops)
@pytest.mark.parametrize("k", k_list)
@pytest.mark.parametrize("scalars_dtype", [torch.float32, torch.bfloat16, torch.long, torch.int], ids=format_tc)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_compound_foreach_tensor_inplace(op, k, scalars_dtype):
    if _is_python_2_4():
        pytest.skip(reason="aten::<op>.out is not yet supported on HPU")
    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, tensor2_cpu, tensor2_hpu = create_compound_foreach_tensors(
        k, op == torch._foreach_addcdiv_, True, scalars_dtype.is_floating_point
    )

    scalars_cpu = (torch.rand(size=(k,)) * 10).to(scalars_dtype)
    if _is_python_2_4():
        scalars_hpu = scalars_cpu
    else:
        scalars_hpu = scalars_cpu.to("hpu")

    if verbose:
        print("Scalars tensor:", scalars_cpu)

    op(self_cpu, tensor1_cpu, tensor2_cpu, scalars=scalars_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, tensor1_hpu, tensor2_hpu, scalars=scalars_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype, True)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", compound_foreach_inplace_ops)
@pytest.mark.parametrize("k", k_list)
@pytest.mark.parametrize("value", scalar_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_compound_foreach_scalar_inplace(op, k, value):
    if _is_python_2_4():
        pytest.skip(reason="aten::<op>.out is not yet supported on HPU")
    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, tensor2_cpu, tensor2_hpu = create_compound_foreach_tensors(
        k, op == torch._foreach_addcdiv_, True, isinstance(value, float)
    )

    if verbose:
        print("Value:", value)

    op(self_cpu, tensor1_cpu, tensor2_cpu, value=value)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, tensor1_hpu, tensor2_hpu, value=value)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype, True)
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("op", compound_foreach_inplace_ops)
@pytest.mark.parametrize("k", k_list)
@pytest.mark.skipif(is_pytest_mode_compile(), reason="Required fallback to eager")
def test_compound_foreach_scalarlist_inplace(op, k):
    if _is_python_2_4():
        pytest.skip(reason="aten::<op>.out is not yet supported on HPU")
    scalars = random.choices(scalar_list, k=k)

    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, tensor2_cpu, tensor2_hpu = create_compound_foreach_tensors(
        k, op == torch._foreach_addcdiv_, True, [isinstance(scalar, float) for scalar in scalars]
    )

    if verbose:
        print("Scalars:", scalars)

    op(self_cpu, tensor1_cpu, tensor2_cpu, scalars=scalars)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, tensor1_hpu, tensor2_hpu, scalars=scalars)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype, True)
    torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


lerp_dtypes = [torch.float, torch.bfloat16]
if not is_gaudi1:
    lerp_dtypes.append(torch.float16)


def generate_foreach_lerp_input(k, optional_scalar):
    indexes = [random.randint(0, len(self_shapes_pull) - 1) for _ in range(k)]
    self_shapes = [self_shapes_pull[idx] for idx in indexes]
    tensor1_shapes = [other_shapes_pull[idx] for idx in indexes]
    dtypes = random.choices(lerp_dtypes, k=k)

    self_cpu, self_hpu = generate_tensor_list(self_shapes, dtypes)
    tensor1_cpu, tensor1_hpu = generate_tensor_list(tensor1_shapes, dtypes)

    if optional_scalar:
        weight_cpu = weight_hpu = optional_scalar
    else:
        weight_cpu, weight_hpu = generate_tensor_list(tensor1_shapes, dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Dtypes:", dtypes)
        print("Tensor1 shapes:", tensor1_shapes)
        print("Weight:", weight_cpu)

    return self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, weight_cpu, weight_hpu


@pytest.mark.parametrize("k", k_list)
@pytest.mark.parametrize("optional_scalar", scalar_list + [None])
def test_foreach_lerp(k, optional_scalar):
    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, weight_cpu, weight_hpu = generate_foreach_lerp_input(
        k, optional_scalar
    )

    op = torch._foreach_lerp
    results_cpu = op(self_cpu, tensor1_cpu, weight_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op(self_hpu, tensor1_hpu, weight_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        if self_cpu[i].dtype == torch.bfloat16:
            rtol, atol = 0.024, 4e-3
        if self_cpu[i].dtype == torch.float16:
            rtol, atol = 1e-4, 1e-2
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("k", k_list)
@pytest.mark.parametrize("optional_scalar", scalar_list + [None])
def test_foreach_lerp_inplace(k, optional_scalar):
    self_cpu, self_hpu, tensor1_cpu, tensor1_hpu, weight_cpu, weight_hpu = generate_foreach_lerp_input(
        k, optional_scalar
    )

    op = torch._foreach_lerp_
    op(self_cpu, tensor1_cpu, weight_cpu)
    op = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op(self_hpu, tensor1_hpu, weight_hpu)

    for i in range(k):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        if self_cpu[i].dtype == torch.bfloat16:
            rtol, atol = 0.024, 4e-3
        if self_cpu[i].dtype == torch.float16:
            rtol, atol = 1e-4, 1e-2
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "op",
    [
        torch._foreach_abs,
        torch._foreach_acos,
        torch._foreach_asin,
        torch._foreach_atan,
        torch._foreach_ceil,
        torch._foreach_cos,
        torch._foreach_cosh,
        torch._foreach_erf,
        torch._foreach_erfc,
        torch._foreach_exp,
        torch._foreach_expm1,
        torch._foreach_floor,
        torch._foreach_frac,
        torch._foreach_neg,
        torch._foreach_lgamma,
        torch._foreach_log,
        torch._foreach_log10,
        torch._foreach_log1p,
        torch._foreach_log2,
        torch._foreach_tan,
        torch._foreach_tanh,
        torch._foreach_round,
        torch._foreach_reciprocal,
        torch._foreach_sign,
        torch._foreach_sigmoid,
        torch._foreach_sin,
        torch._foreach_sinh,
        torch._foreach_sqrt,
        torch._foreach_trunc,
    ],
)
def test_foreach_unary(op):
    if op == torch._foreach_lgamma and is_gaudi1():
        pytest.skip(reason="foreach_lgamma is unsupported for Gaudi")

    self_shapes = random.choices(self_shapes_pull, k=len(dtypes))
    self_dtypes = dtypes[:]

    if _is_python_2_4():
        op_to_skip = [torch._foreach_round, torch._foreach_log2, torch._foreach_lgamma]
        if op in op_to_skip:
            pytest.skip(
                reason="aten::op.out is not yet supported on HPU. Guid op_fwd has incompatible input or output data types"
            )
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)

    for i in range(len(self_dtypes)):
        if not self_dtypes[i].is_floating_point and op == torch._foreach_frac:
            self_dtypes[i] = torch.float32

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)

    results_cpu = op(self_cpu)
    op_hpu = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    results_hpu = op_hpu(self_hpu)

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, results_cpu[i].dtype)
        if (
            op in [torch._foreach_log, torch._foreach_log2, torch._foreach_sigmoid, torch._foreach_erfc]
            and self_cpu[i].dtype == torch.bfloat16
        ):
            rtol, atol = 1e-5, 0.02
        if op in [torch._foreach_log, torch._foreach_log2, torch._foreach_log1p] and self_cpu[i].dtype == torch.float16:
            rtol, atol = 3e-5, 0.002
        torch.testing.assert_close(results_cpu[i], results_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)


integer_foreach_unary_inplace_ops = [
    torch._foreach_abs_,
    torch._foreach_ceil_,
    torch._foreach_floor_,
    torch._foreach_neg_,
    torch._foreach_round_,
    torch._foreach_trunc_,
    torch._foreach_sign_,
    torch._foreach_zero_,
]

non_integer_foreach_unary_inplace_ops = [
    torch._foreach_acos_,
    torch._foreach_asin_,
    torch._foreach_atan_,
    torch._foreach_cos_,
    torch._foreach_cosh_,
    torch._foreach_erf_,
    torch._foreach_erfc_,
    torch._foreach_exp_,
    torch._foreach_expm1_,
    torch._foreach_frac_,
    torch._foreach_lgamma_,
    torch._foreach_log_,
    torch._foreach_log10_,
    torch._foreach_log1p_,
    torch._foreach_log2_,
    torch._foreach_tan_,
    torch._foreach_tanh_,
    torch._foreach_reciprocal_,
    torch._foreach_sigmoid_,
    torch._foreach_sin_,
    torch._foreach_sinh_,
    torch._foreach_sqrt_,
]


@pytest.mark.parametrize("op", integer_foreach_unary_inplace_ops + non_integer_foreach_unary_inplace_ops)
def test_foreach_unary_inplace(op):
    if op == torch._foreach_lgamma_ and is_gaudi1():
        pytest.skip(reason="foreach_lgamma is unsupported for Gaudi")
    self_shapes = random.choices(self_shapes_pull, k=len(dtypes))
    self_dtypes = dtypes[:]

    if _is_python_2_4():
        op_to_skip = [torch._foreach_round_, torch._foreach_log2_, torch._foreach_lgamma_]
        if op in op_to_skip:
            pytest.skip(
                reason="aten::op.out is not yet supported on HPU. Guid op_fwd has incompatible input or output data types"
            )
        _remove_not_supported_dtypes_on_hpu(op, self_dtypes)

    for i in range(len(self_dtypes)):
        if not self_dtypes[i].is_floating_point and op in non_integer_foreach_unary_inplace_ops:
            self_dtypes[i] = torch.float32

    self_cpu, self_hpu = generate_tensor_list(self_shapes, self_dtypes)

    if verbose:
        print("Self shapes:", self_shapes)
        print("Self dtypes:", self_dtypes)

    op(self_cpu)
    op_hpu = torch.compile(op, backend="hpu_backend") if is_pytest_mode_compile() else op
    op_hpu(self_hpu)

    for i in range(len(self_dtypes)):
        rtol, atol = get_tolerance(op, self_cpu[i].dtype)
        if (
            op in [torch._foreach_log_, torch._foreach_log2_, torch._foreach_sigmoid_, torch._foreach_erfc_]
            and self_cpu[i].dtype == torch.bfloat16
        ):
            rtol, atol = 1e-5, 0.02
        if (
            op in [torch._foreach_log_, torch._foreach_log2_, torch._foreach_log1p_]
            and self_cpu[i].dtype == torch.float16
        ):
            rtol, atol = 3e-5, 0.002
        torch.testing.assert_close(self_cpu[i], self_hpu[i].cpu(), equal_nan=True, rtol=rtol, atol=atol)
