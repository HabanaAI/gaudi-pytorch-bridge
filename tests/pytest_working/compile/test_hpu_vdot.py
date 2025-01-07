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
