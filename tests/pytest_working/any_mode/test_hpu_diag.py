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


@pytest.mark.parametrize("shape_and_diag", [((24,), 0), ((8, 8), 0), ((8, 8), 1)], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int], ids=format_tc)
def test_hpu_diag(shape_and_diag, dtype):
    if pytest.mode == "compile" and shape_and_diag in [((8, 8), 0), ((8, 8), 1)]:
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")

    def fn(input):
        return torch.diag(input, diagonal=diagonal)

    shape, diagonal = shape_and_diag
    if dtype == torch.int:
        cpu_input = torch.randint(low=-128, high=127, size=shape, dtype=dtype)
    else:
        cpu_input = torch.rand(shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input).cpu()

    assert torch.equal(cpu_output, hpu_output)
