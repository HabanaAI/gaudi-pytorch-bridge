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


@pytest.mark.parametrize("shape_and_dim", [((2, 3, 4), -1), ((2, 3, 4), -2)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
def test_hpu_gather(shape_and_dim, dtype):
    def fn(input, indices):
        return torch.gather(input, dim, indices)

    shape, dim = shape_and_dim
    max_index = shape[-1] - 1
    cpu_input = (
        torch.rand(shape, dtype=dtype)
        if dtype.is_floating_point
        else torch.randint(low=-127, high=127, size=shape, dtype=dtype)
    )
    hpu_input = cpu_input.to("hpu")
    cpu_indices = torch.randint(low=0, high=max_index, size=shape, dtype=torch.int64)
    hpu_indices = cpu_indices.to("hpu")

    torch._dynamo.reset()
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input, cpu_indices)
    hpu_output = hpu_wrapped_fn(hpu_input, hpu_indices).cpu()
    assert torch.equal(cpu_output, hpu_output)
