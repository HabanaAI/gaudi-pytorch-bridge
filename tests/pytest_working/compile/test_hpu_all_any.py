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
from test_utils import format_tc, is_gaudi1
import habana_frameworks.torch.dynamo.compile_backend


shapes = [[1], [2,3,4], [4,2]]
use_out = [True, False]
dtypes = ["bfloat16", "float", "int"]
if not is_gaudi1():
    dtypes.append("float16")
    dtypes.append("short")


def fn(input_tensor, use_out, output_device, op):
    if use_out:
        output_tensor = torch.tensor(1).to(output_device)
        op(input_tensor, out=output_tensor)
        return output_tensor

    return op(input_tensor)


def check(cpu_input, use_out, op):
    hpu_input = cpu_input.to("hpu")

    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input, use_out, "cpu", op)
    torch._dynamo.reset()
    hpu_output = hpu_compiled_fn(hpu_input, use_out, "hpu", op).cpu()

    assert torch.equal(cpu_output, hpu_output)


@pytest.mark.parametrize("use_out", use_out)
@pytest.mark.parametrize("shape", shapes, ids=format_tc)
def test_hpu_all(use_out, shape):
    cpu_input = torch.randint(size=shape, low=0, high=2, dtype=torch.bool)
    check(cpu_input, use_out, torch.all)


@pytest.mark.parametrize("use_out", use_out)
@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes)
def test_hpu_any(use_out, shape, dtype):
    dtype = getattr(torch, dtype)

    if dtype in (torch.int, torch.short):
        cpu_input = torch.randint(size=shape, low=0, high=2, dtype=dtype)
    else:
        cpu_input = torch.rand(shape, dtype=dtype)

    check(cpu_input, use_out, torch.any)