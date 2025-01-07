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


@pytest.mark.parametrize("shape", [[2, 2, 4], [4, 6, 4, 2, 6]], ids=format_tc)
@pytest.mark.parametrize("dim", [-1, 0, 2])
@pytest.mark.parametrize("backward", [False, True])
def test_hpu_glu(shape, dim, backward):
    if backward:
        pytest.xfail("SW-155324")

    def fn_fwd(input_tensor, dim):
        glu = torch.nn.GLU(dim=dim)
        return glu(input_tensor)

    def fn_bwd(input_tensor, dim):
        glu = torch.nn.GLU(dim=dim)
        output = glu(input_tensor)
        grad = torch.ones_like(output)
        output.backward(grad)
        return input_tensor.grad

    cpu_input = torch.rand(shape, dtype=torch.float)
    hpu_input = cpu_input.to("hpu")

    if backward:
        cpu_input.requires_grad = True
        hpu_input.requires_grad = True
        fn = fn_bwd
    else:
        fn = fn_fwd

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, dim)
    hpu_output = hpu_compiled_fn(hpu_input, dim).cpu()

    assert torch.allclose(cpu_output, hpu_output)
