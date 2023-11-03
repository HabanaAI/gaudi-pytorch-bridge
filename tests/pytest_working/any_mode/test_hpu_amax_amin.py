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
import habana_frameworks.torch.dynamo.compile_backend

@pytest.mark.parametrize("op", [torch.amin, torch.amax])
@pytest.mark.parametrize("shapes", [(3, 4, 5, 6), (2, 3, 5, 4)])
@pytest.mark.parametrize("dim", [-4, -3, -2, -1, 0, 1, 2 ,3])
@pytest.mark.parametrize("dtype", ["float"])
def test_hpu_amax_amin(op, shapes, dim, dtype):
    def fn(input, dim):
        return op(input, dim)

    cpu_input = torch.randn(shapes, dtype=getattr(torch, dtype))
    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else fn
    cpu_output = cpu_wrapped_fn(cpu_input, dim)
    hpu_output = hpu_wrapped_fn(hpu_input, dim).cpu()
    assert torch.allclose(cpu_output, hpu_output)
