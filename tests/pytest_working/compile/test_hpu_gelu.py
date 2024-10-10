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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch


@pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 3, 4), (4, 8, 16, 32)])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gelu(shape, approximate, dtype):
    def fn(input):
        return torch.nn.functional.gelu(input, approximate=approximate)

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    hpu_result = hpu_compiled_fn(hpu_input).cpu()
    cpu_result = fn(cpu_input)
    rtol = 1e-02 if dtype == torch.bfloat16 else 1e-04
    atol = 1e-04
    assert torch.allclose(cpu_result, hpu_result, rtol, atol)
