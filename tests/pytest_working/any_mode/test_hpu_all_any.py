###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
from test_utils import compare_tensors, format_tc, is_gaudi1, is_pytest_mode_compile

zero_size_shapes = [[0], [0, 1], [0, 1, 2]]

# input for torch.all
input_all_op = [[], [1], [2, 3, 4], [4, 2]]  # shapes

# input for torch.any
input_any_op = [
    [[], None],
    [[1], None],  # [shapes, dim]
    [[2, 3, 4], None],
    [[4, 2], None],
    [[], 0],
    [[1], 0],
    [[2, 3, 4], 2],
    [[4, 2], 1],
    [[3, 2], -1],
    [[2, 3, 4], (1, 2)],
    [[4, 2], (1, 0)],
    [[3, 2], (-1, 0)],
]


use_out = [True, False]
dtypes = [torch.bfloat16, torch.float, torch.int]
if not is_gaudi1():
    dtypes.append(torch.float16)
    dtypes.append(torch.short)


def fn(input_tensor, use_out, output_device, op, dim):
    if use_out:
        output_tensor = torch.tensor(True).to(output_device)
        if dim == None:
            op(input_tensor, out=output_tensor)
        else:
            op(input_tensor, dim=dim, out=output_tensor)
        return output_tensor

    if dim == None:
        return op(input_tensor)
    else:
        return op(input_tensor, dim=dim)


def check(cpu_input, use_out, op, dim):
    hpu_input = cpu_input.to("hpu")
    hpu_fn = fn
    cpu_output = fn(cpu_input, use_out, "cpu", op, dim)
    if is_pytest_mode_compile():
        torch._dynamo.reset()
        hpu_fn = torch.compile(fn, backend="hpu_backend")
    hpu_output = hpu_fn(hpu_input, use_out, "hpu", op, dim).cpu()
    compare_tensors([hpu_output], [cpu_output], atol=0, rtol=0)


@pytest.mark.parametrize("use_out", use_out)
@pytest.mark.parametrize("shape", input_all_op, ids=format_tc)
def test_hpu_all(use_out, shape):
    cpu_input = torch.randint(size=shape, low=0, high=2, dtype=torch.bool)
    check(cpu_input, use_out, torch.all, None)


@pytest.mark.parametrize("use_out", use_out)
@pytest.mark.parametrize("shape", zero_size_shapes, ids=format_tc)
def test_hpu_all_zero_size(use_out, shape):
    cpu_input = torch.empty(shape, dtype=torch.bool)
    check(cpu_input, use_out, torch.all, None)


@pytest.mark.parametrize("use_out", use_out)
@pytest.mark.parametrize("input", input_any_op, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_any(use_out, input, dtype):
    shape = input[0]
    dim = input[1]

    if dtype in (torch.int, torch.short):
        cpu_input = torch.randint(size=shape, low=0, high=2, dtype=dtype)
    else:
        cpu_input = torch.rand(shape, dtype=dtype)

    check(cpu_input, use_out, torch.any, dim)
