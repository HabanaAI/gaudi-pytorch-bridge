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
from test_utils import format_tc


@pytest.mark.parametrize("input_shape", [(1, 2, 3, 4), (2, 4, 4, 2)], ids=format_tc)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_grid_sampler_2d(input_shape, align_corners, dtype):
    def fn(input, grid):
        return torch.grid_sampler_2d(input, grid, 0, 0, align_corners)

    N, _, H, W = input_shape
    grid_shape = (N, H, W, 2)
    cpu_input = torch.rand(input_shape, dtype=dtype)
    cpu_grid = torch.rand(grid_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    hpu_grid = cpu_grid.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input, cpu_grid)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_grid).cpu()

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output, rtol=tol, atol=tol)
