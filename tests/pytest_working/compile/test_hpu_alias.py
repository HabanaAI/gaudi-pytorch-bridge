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
import pytest
import torch


@pytest.mark.parametrize("shape", [(1,), (1, 2), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int8, torch.int32])
def test_alias(shape, dtype):
    def fn(input):
        return torch.ops.aten.alias(input)

    torch._dynamo.reset()
    cpu_input = (
        torch.randn(shape, dtype=dtype)
        if dtype.is_floating_point
        else torch.randint(low=-128, high=127, size=shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()

    assert torch.equal(hpu_output, cpu_output)


# add cases for SW-163523
@pytest.mark.parametrize("shape", [(1, 2), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int8, torch.int32])
def test_alias_with_view_input(shape, dtype):
    def fn(input):
        input = torch.ops.aten.transpose(input, 0, 1)
        return torch.ops.aten.alias(input)

    torch._dynamo.reset()
    cpu_input = (
        torch.randn(shape, dtype=dtype)
        if dtype.is_floating_point
        else torch.randint(low=-128, high=127, size=shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input)
    hpu_output = hpu_compiled_fn(hpu_input).cpu()

    assert torch.equal(hpu_output, cpu_output)
