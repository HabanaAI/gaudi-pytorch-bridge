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


@pytest.mark.parametrize("shapes", [([3, 2], [2], [3]), ([9, 4], [4], [9])])
@pytest.mark.parametrize("alpha", [0.5])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_inplace_addr_with_view_input(shapes, alpha, beta, dtype):
    def fn(input, vec1, vec2):
        input = torch.transpose(input, 0, 1)
        return input.addr_(vec1, vec2, alpha=alpha, beta=beta)

    input_shape, mat_shape, vec_shape = shapes
    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_vec1 = torch.rand(mat_shape, dtype=dtype)
    hpu_vec1 = cpu_vec1.to("hpu")
    cpu_vec2 = torch.rand(vec_shape, dtype=dtype)
    hpu_vec2 = cpu_vec2.to("hpu")
    torch._dynamo.reset()
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = cpu_compiled_fn(cpu_input, cpu_vec1, cpu_vec2)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_vec1, hpu_vec2).cpu()
    assert torch.allclose(cpu_output, hpu_output)
