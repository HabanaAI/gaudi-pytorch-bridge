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


@pytest.mark.parametrize("shape", [(4), (2, 2), (2, 3, 4)], ids=format_tc)
@pytest.mark.parametrize("bwd", [True, False])
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
def test_hpu_logsigmoid(shape, bwd, dtype):
    def forward(input):
        return torch.ops.aten.log_sigmoid_forward(input)[0]

    def backward(input):
        fwd_result, buffer = torch.ops.aten.log_sigmoid_forward(input)
        grad = torch.ones_like(fwd_result)
        return torch.ops.aten.log_sigmoid_backward(grad, input, buffer)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    wrapped_fn = backward if bwd else forward

    torch._dynamo.reset()
    hpu_wrapped_fn = torch.compile(wrapped_fn, backend="hpu_backend") if pytest.mode == "compile" else wrapped_fn

    cpu_output = wrapped_fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()
    assert torch.allclose(cpu_output, hpu_output)
