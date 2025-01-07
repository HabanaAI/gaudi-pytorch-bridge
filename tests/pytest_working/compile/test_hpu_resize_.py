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


@pytest.mark.parametrize("dest_shape", [[1, 2], [4, 1, 2]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_resize_(dest_shape, dtype):
    def fn(input, dest_shape):
        input.resize_(dest_shape)

    cpu_input = torch.rand([2, 3, 4], dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    fn(cpu_input, dest_shape)
    hpu_compiled_fn(hpu_input, dest_shape)
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-7
    assert torch.allclose(cpu_input, hpu_input.cpu(), rtol=rtol)
