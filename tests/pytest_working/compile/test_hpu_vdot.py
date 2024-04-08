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


@pytest.mark.parametrize("shapes", [2, 10])
@pytest.mark.parametrize("dtype", ["float", "bfloat16"])
def test_hpu_vdot(shapes, dtype):
    def fn(a, b):
        return torch.vdot(a, b)

    cpu_a = torch.rand(shapes, dtype=getattr(torch, dtype))
    hpu_a = cpu_a.to("hpu")
    cpu_b = torch.rand(shapes, dtype=getattr(torch, dtype))
    hpu_b = cpu_b.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    torch._dynamo.reset()

    cpu_output = fn(cpu_a, cpu_b)
    hpu_output = hpu_compiled_fn(hpu_a, hpu_b).cpu()
    torch.allclose(cpu_output, hpu_output)
