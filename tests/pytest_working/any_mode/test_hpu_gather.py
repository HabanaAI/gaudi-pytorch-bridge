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
import pytest
import torch


@pytest.mark.parametrize("shape_and_dim", [((2, 3, 4), -1), ((2, 3, 4), -2)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
def test_hpu_gather(shape_and_dim, dtype):
    def fn(input, indices):
        return torch.gather(input, dim, indices)

    shape, dim = shape_and_dim
    max_index = shape[-1] - 1
    cpu_input = (
        torch.rand(shape, dtype=dtype)
        if dtype.is_floating_point
        else torch.randint(low=-127, high=127, size=shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    cpu_indices = torch.randint(low=0, high=max_index, size=shape, dtype=torch.int64)
    hpu_indices = cpu_indices.to("hpu")

    torch._dynamo.reset()
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input, cpu_indices)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_indices).cpu()
    assert torch.equal(cpu_output, hpu_output)
