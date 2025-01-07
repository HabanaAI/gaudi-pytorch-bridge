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
import torch.nn.functional as F
from test_utils import format_tc

shapes_data = [
    ((0, 6), (3, 2), 6),
    ((0, 8), (2, 2, 2), 3),
    ((0, 16), (2, 2, 2, 2), 5),
]


@pytest.mark.parametrize("classes", [6, 50])
# If INT64 is not enabled in HPU, Long maps to i32 in TPC (in eager mode)
@pytest.mark.parametrize("dtype", ["long"])
@pytest.mark.parametrize("shape", shapes_data, ids=format_tc)
def test_hpu_one_hot(shape, classes, dtype):
    def fn(input, classes):
        return F.one_hot(input, num_classes=classes)

    arange, view, mod = shape
    cpu_input = torch.arange(*arange, dtype=getattr(torch, dtype)).view(*view) % mod
    hpu_input = cpu_input.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn
    torch._dynamo.reset()

    cpu_output = fn(cpu_input, classes)
    hpu_output = hpu_compiled_fn(hpu_input, classes).cpu()

    assert torch.equal(cpu_output, hpu_output)
