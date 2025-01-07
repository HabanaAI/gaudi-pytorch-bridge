###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
