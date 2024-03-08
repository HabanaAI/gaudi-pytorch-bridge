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
import habana_frameworks.torch.dynamo.compile_backend
import habana_frameworks.torch.utils.experimental as htexp
import pytest
import torch
from test_utils import format_tc, is_gaudi1

dtypes = [torch.bfloat16, torch.float]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("input_shape", [[2, 2, 3], [4, 2, 5, 2], [2, 1, 3, 3, 2]])
@pytest.mark.parametrize("other_scalar", [None, 1.3, 2.123, 5.947812, 13.13541])
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_div_tensor_scalar_mode(input_shape, other_scalar, rounding_mode, dtype):
    def fn(input, other, rounding_mode):
        return torch.div(input, other, rounding_mode=rounding_mode)

    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    if other_scalar is None:
        cpu_other = torch.rand(input_shape, dtype=dtype)
        hpu_other = cpu_other.to("hpu")
    else:
        cpu_other = other_scalar
        hpu_other = other_scalar

    cpu_compiled_fn = fn
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_compiled_fn(cpu_input, cpu_other, rounding_mode)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_other, rounding_mode).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("input_scalar", [1.3, 2.123, 5.947812, 13.13541])
@pytest.mark.parametrize("other_shape", [[1]])
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_div_scalar_tensor_mode(input_scalar, other_shape, rounding_mode, dtype):
    def fn(input, other, rounding_mode):
        return torch.div(input, other, rounding_mode=rounding_mode)

    cpu_input = input_scalar
    hpu_input = input_scalar

    cpu_other = torch.rand(other_shape, dtype=dtype)
    hpu_other = cpu_other.to("hpu")

    cpu_compiled_fn = fn
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_compiled_fn(cpu_input, cpu_other, rounding_mode)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_other, rounding_mode).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("input_shape", [[2, 2, 3], [4, 2, 5, 2], [2, 1, 3, 3, 2]])
@pytest.mark.parametrize("other_scalar", [None, 1.3, 2.123, 5.947812, 13.13541])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_floor_divide_tensor_scalar_mode(input_shape, other_scalar, dtype):
    def fn(input, other):
        return torch.floor_divide(input, other)

    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    if other_scalar == None:
        cpu_other = torch.rand(input_shape, dtype=dtype)
        hpu_other = cpu_other.to("hpu")
    else:
        cpu_other = other_scalar
        hpu_other = other_scalar

    cpu_compiled_fn = fn
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_compiled_fn(cpu_input, cpu_other)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_other).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("input_scalar", [1.3, 2.123, 5.947812, 13.13541])
@pytest.mark.parametrize("other_shape", [[1]])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_floor_divide_scalar_tensor_mode(input_scalar, other_shape, dtype):
    def fn(input, other):
        return torch.floor_divide(input, other)

    cpu_input = input_scalar
    hpu_input = input_scalar

    cpu_other = torch.rand(other_shape, dtype=dtype)
    hpu_other = cpu_other.to("hpu")

    cpu_compiled_fn = fn
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = cpu_compiled_fn(cpu_input, cpu_other)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_other).cpu()
    assert torch.allclose(cpu_output, hpu_output)
