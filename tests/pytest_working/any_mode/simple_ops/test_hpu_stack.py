###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
from test_utils import compile_function_if_compile_mode, format_tc, use_eager_fallback

dtypes = [torch.bfloat16, torch.float, torch.float16]


@pytest.mark.parametrize("input_tensor", ([], [1, 2], [[2, 2], [2, 3]]))
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_stack(input_tensor, dtype):
    def fn(input):
        return torch.stack([input, input])

    cpu_input = torch.tensor(input_tensor, dtype=dtype)
    cpu_output = fn(cpu_input)

    hpu_input = cpu_input.to("hpu")
    fn = compile_function_if_compile_mode(fn)
    with use_eager_fallback():
        hpu_output = fn(hpu_input)

    assert torch.allclose(cpu_output, hpu_output.cpu())
