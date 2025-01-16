###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
from test_utils import compile_function_if_compile_mode


@pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 3, 4), (4, 8, 16, 32)])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gelu(shape, approximate, dtype):
    def fn(input):
        return torch.nn.functional.gelu(input, approximate=approximate)

    hpu_compiled_fn = compile_function_if_compile_mode(fn)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    hpu_result = hpu_compiled_fn(hpu_input).cpu()
    cpu_result = fn(cpu_input)
    rtol = 1e-02 if dtype == torch.bfloat16 else 1e-04
    atol = 1e-04
    assert torch.allclose(cpu_result, hpu_result, rtol, atol)
