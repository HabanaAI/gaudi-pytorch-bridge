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
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import format_tc, is_gaudi1

params = [
    ([8, 2, 3], [0, 2]),
    ((4, 4, 4, 2, 2), []),
    ((1, 9, 5), 2),
    ((2, 4), [-1]),
]


dtypes = [
    torch.bfloat16,
    torch.float,
    torch.int,
    torch.short,
    torch.bool,
]

if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("shape, dim", params, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_hpu_count_nonzero(shape, dim, dtype):
    torch._dynamo.reset()

    def fn(input, dim):
        return torch.count_nonzero(input, dim)

    cpu_input = torch.randint(low=0, high=2, size=shape, dtype=dtype)
    if dtype in (torch.bfloat16, torch.float16, torch.float):
        cpu_input *= torch.rand(shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu")

    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = cpu_compiled_fn(cpu_input, dim)
    hpu_output = hpu_compiled_fn(hpu_input, dim).to("cpu")

    assert torch.equal(cpu_output, hpu_output)
