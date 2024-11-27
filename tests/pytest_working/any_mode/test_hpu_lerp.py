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

import pytest
import torch
from test_utils import format_tc

dtypes = [torch.bfloat16, torch.float, torch.int]


@pytest.mark.parametrize("scalar_weight", [False, True])
@pytest.mark.parametrize("shape", [[2, 2, 4], [4, 6, 4, 2, 6]], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_lerp(shape, scalar_weight, dtype):
    def fn(start, end, weight):
        return torch.lerp(start, end, weight)

    if dtype == torch.int:
        cpu_start = torch.randint(size=shape, low=0, high=15, dtype=dtype)
        cpu_end = torch.randint(size=shape, low=0, high=15, dtype=dtype)
    else:
        cpu_start = torch.rand(shape, dtype=dtype)
        cpu_end = torch.rand(shape, dtype=dtype)

    hpu_start = cpu_start.to("hpu")
    hpu_end = cpu_end.to("hpu")

    if scalar_weight:
        cpu_weight = 5 if dtype == torch.int else 2.5647382
        hpu_weight = cpu_weight
    else:
        if dtype == torch.int:
            cpu_weight = torch.randint(size=shape, low=0, high=20, dtype=dtype)
        else:
            cpu_weight = torch.rand(shape, dtype=dtype)
        hpu_weight = cpu_weight.to("hpu")

    if dtype == torch.int:
        cpu_start = cpu_start.to(torch.float32)
        cpu_end = cpu_end.to(torch.float32)
        cpu_weight = cpu_weight if scalar_weight else cpu_weight.to(torch.float32)

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_start, cpu_end, cpu_weight)
    hpu_output = hpu_compiled_fn(hpu_start, hpu_end, hpu_weight).cpu()

    if dtype == torch.int:
        cpu_output = cpu_output.to(torch.int)

    tol = 1e-5 if dtype != torch.bfloat16 else 1e-1

    assert torch.allclose(cpu_output, hpu_output, atol=tol, rtol=tol)
