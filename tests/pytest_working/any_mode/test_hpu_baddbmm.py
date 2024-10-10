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


@pytest.mark.parametrize("alpha", [0.5, 1.0])
@pytest.mark.parametrize("beta", [0.5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_baddbmm(alpha, beta, dtype):
    def fn(input, batch1, batch2):
        return torch.baddbmm(input, batch1, batch2, alpha=alpha, beta=beta)

    B, M, N, P = 4, 3, 3, 4
    cpu_input = torch.rand((1, N, P), dtype=dtype)
    cpu_batch1 = torch.rand((B, N, M), dtype=dtype)
    cpu_batch2 = torch.rand((B, M, P), dtype=dtype)

    hpu_input = cpu_input.to("hpu")
    hpu_batch1 = cpu_batch1.to("hpu")
    hpu_batch2 = cpu_batch2.to("hpu")

    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input, cpu_batch1, cpu_batch2)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_batch1, hpu_batch2).cpu()

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output, rtol=tol, atol=tol)
