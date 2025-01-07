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


@pytest.mark.parametrize("shapes", [([2, 3], [3])])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float, torch.short, torch.int])
def test_hpu_mv_ops(shapes, dtype):
    if pytest.mode in ["lazy", "eager"] and dtype in [torch.short, torch.int]:
        pytest.skip(reason=f"aten::addmv.out for {dtype} is not yet supported on HPU")

    def fn(mat, vec):
        return torch.mv(mat, vec)

    mat_shape, vec_shape = shapes
    if dtype in [torch.bfloat16, torch.float]:
        cpu_mat = torch.rand(mat_shape, dtype=dtype)
        cpu_vec = torch.rand(vec_shape, dtype=dtype)
    else:
        cpu_mat = torch.randint(0, 10, mat_shape, dtype=dtype)
        cpu_vec = torch.randint(0, 10, vec_shape, dtype=dtype)

    hpu_mat = cpu_mat.to("hpu")
    hpu_vec = cpu_vec.to("hpu")
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if (pytest.mode == "compile") else fn

    cpu_output = fn(cpu_mat, cpu_vec)
    hpu_output = hpu_wrapped_fn(hpu_mat, hpu_vec).cpu()
    assert torch.allclose(cpu_output, hpu_output, rtol=1e-2, atol=1e-2)
