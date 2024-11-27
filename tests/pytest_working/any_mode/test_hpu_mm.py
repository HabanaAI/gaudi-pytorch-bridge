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


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("M", [4, 5])
@pytest.mark.parametrize("P", [3, 2])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_hpu_mm(N, M, P, dtype):
    def fn(mat1, mat2):
        return torch.mm(mat1, mat2)

    shape_mat1 = (N, M)
    shape_mat2 = (M, P)
    cpu_mat1 = torch.rand(shape_mat1, dtype=dtype)
    hpu_mat1 = cpu_mat1.to("hpu")
    cpu_mat2 = torch.rand(shape_mat2, dtype=dtype)
    hpu_mat2 = cpu_mat2.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_mat1, cpu_mat2)
    hpu_output = hpu_wrapped_fn(hpu_mat1, hpu_mat2).cpu()
    assert torch.allclose(cpu_output, hpu_output)
