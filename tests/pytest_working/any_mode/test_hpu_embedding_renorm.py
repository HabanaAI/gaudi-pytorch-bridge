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


@pytest.mark.parametrize("norm_type", [0.0, 1.0, 2.0, float("inf"), float("-inf"), 1.342, 3.423, -4.234])
@pytest.mark.parametrize("max_norm", [-3.093, 1.0, 2.423, 12.234, 200.0])
@pytest.mark.parametrize("shape", [[20, 14]], ids=format_tc)
def test_embedding_renorm(norm_type, max_norm, shape):
    def fn(input, indices):
        return torch.embedding_renorm_(input, indices, max_norm=max_norm, norm_type=norm_type)

    dtype = torch.float

    input_cpu = torch.randn(shape, dtype=dtype)
    indices_cpu = torch.randint(size=[shape[0] // 2], low=0, high=shape[0] - 1)

    input_hpu = input_cpu.to("hpu")
    indices_hpu = indices_cpu.to("hpu")

    compiled_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    result_cpu = fn(input_cpu, indices_cpu)
    result_hpu = compiled_fn(input_hpu, indices_hpu)

    return torch.allclose(result_cpu, result_hpu.cpu())
