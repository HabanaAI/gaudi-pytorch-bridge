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


@pytest.mark.parametrize("shapes", [([2], [2, 3], [3]), ([4], [4, 9], [9])])
@pytest.mark.parametrize("alpha", [0.5, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_addmv(shapes, alpha, beta, dtype):
    def fn(input, mat, vec):
        return torch.addmv(input, mat, vec, alpha=alpha, beta=beta)

    input_shape, mat_shape, vec_shape = shapes
    cpu_input = torch.rand(input_shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    cpu_mat = torch.rand(mat_shape, dtype=dtype)
    hpu_mat = cpu_mat.to("hpu")
    cpu_vec = torch.rand(vec_shape, dtype=dtype)
    hpu_vec = cpu_vec.to("hpu")
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = fn(cpu_input, cpu_mat, cpu_vec)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_mat, hpu_vec).cpu()
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-7
    assert torch.allclose(cpu_output, hpu_output, rtol=rtol)
