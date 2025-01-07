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


@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("m", [4, None])
@pytest.mark.parametrize("dtype", [torch.float, torch.int32], ids=format_tc)
def test_hpu_eye(n, m, dtype):
    def fn(output):
        if m is None:
            torch.eye(n, out=output)
        else:
            torch.eye(n, m, out=output)

    shape = (n, n) if m is None else (n, m)
    cpu_output = torch.empty(shape, dtype=dtype)
    hpu_output = cpu_output.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    fn(cpu_output)
    hpu_wrapped_fn(hpu_output)

    assert torch.equal(cpu_output, hpu_output.cpu())
