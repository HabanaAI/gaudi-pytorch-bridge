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
from test_utils import compare_tensors, format_tc, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.float, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.half)


@pytest.mark.parametrize("shapes", [([1, 3], [2], [3]), ([4, 9], [4], [9])], ids=format_tc)
@pytest.mark.parametrize("alpha", [0, 1, 2.2])
@pytest.mark.parametrize("beta", [0, 1, 2.2])
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_addr(shapes, alpha, beta, dtype):
    def fn(input, vec1, vec2):
        return torch.addr(input, vec1, vec2, alpha=alpha, beta=beta)

    input_shape, mat_shape, vec_shape = shapes

    cpu_input = torch.rand(input_shape, dtype=dtype)
    cpu_input[0][0] = float("inf")
    hpu_input = cpu_input.to("hpu")
    cpu_vec1 = torch.rand(mat_shape, dtype=dtype)
    cpu_vec1[1] = float("nan")
    hpu_vec1 = cpu_vec1.to("hpu")
    cpu_vec2 = torch.rand(vec_shape, dtype=dtype)
    hpu_vec2 = cpu_vec2.to("hpu")

    hpu_fn = torch.compile(fn, backend="hpu_backend") if is_pytest_mode_compile() else fn

    cpu_output = fn(cpu_input, cpu_vec1, cpu_vec2)
    hpu_output = hpu_fn(hpu_input, hpu_vec1, hpu_vec2)

    tol = 1e-6 if dtype == torch.float else 1e-2
    compare_tensors(hpu_output, cpu_output, atol=tol, rtol=tol)
    # Addr is decomposed by default to another OPs, so no check if addr is present in JIT_IR
